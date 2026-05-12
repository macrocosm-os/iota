# The llama 3 implementation was adapted from Sebastian Raschka's one.

# The original implementation is available at:
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).

import torch
import torch.nn as nn
import random
import numpy as np

from subnet.model import gpu_device

seed = 42
torch.manual_seed(seed)
gpu_device.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def init_weights(module):
    """
    Llama3.1 initialization weights.
    This function is used to initialize the weights of the model.
    It is called in the __init__ method of the model.
    It is used to initialize the weights of the model.
    """
    std = 0.02
    if isinstance(module, nn.Linear):
        # Initialize weights with normal distribution (mean=0.0, std=0.02)
        module.weight.data.normal_(0.0, std)
        # Initialize bias with zeros if present
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
        # Initialize embedding weights with normal distribution (mean=0.0, std=0.02)
        module.weight.data.normal_(0.0, std)
        # If padding_idx is specified, set the corresponding embedding vector to zeros
        # This ensures that padding tokens don't contribute to the model's computations
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    elif isinstance(module, nn.RMSNorm):
        module.weight.data.fill_(1.0)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin):
        # Shortcut connection for attention block
        shortcut = x
        # if x has the bottleneck dimension, we need to pass it through the bottleneck decoder
        if x.shape[-1] == self.cfg["bottleneck_dim"] and self.cfg["bottleneck_dim"] != self.cfg["emb_dim"]:
            if not hasattr(self, "bottleneck_decoder"):
                bottleneck_decoder = BottleneckDecoder(
                    self.cfg["bottleneck_dim"], self.cfg["emb_dim"], dtype=self.cfg["dtype"]
                ).to(x.device)
                bottleneck_decoder.apply(init_weights)
                self.add_module("bottleneck_decoder", bottleneck_decoder)

            x = self.bottleneck_decoder(x)

        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]

        if shortcut.shape[-1] == self.cfg["bottleneck_dim"]:
            x_bottleneck_part = x[:, :, : self.cfg["bottleneck_dim"]]
            x_normal_part = x[:, :, self.cfg["bottleneck_dim"] :]
            x = x_bottleneck_part + shortcut  # Add the original input back
            x = torch.cat([x, x_normal_part], dim=-1)
        else:
            x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class BottleneckTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bottleneck_decoder = BottleneckDecoder(cfg["bottleneck_dim"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"],
        )
        self.ff = BottleneckFeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin):
        # Shortcut connection for attention block
        shortcut = x

        # if x has the bottleneck dimension, we need to pass it through the bottleneck decoder
        if shortcut.shape[-1] == self.cfg["bottleneck_dim"]:
            x = self.bottleneck_decoder(x)

        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]

        if shortcut.shape[-1] == self.cfg["bottleneck_dim"]:
            x_bottleneck_part = x[:, :, : self.cfg["bottleneck_dim"]]
            x_normal_part = x[:, :, self.cfg["bottleneck_dim"] :]
            x = x_bottleneck_part + shortcut  # Add the original input back
            x = torch.cat([x, x_normal_part], dim=-1)
        else:
            x = x + shortcut  # Add the original input back

        ## Partial shortcut connection for feed-forward block

        # break x into two parts, one of size cfg["bottleneck_dim"]
        # and the other of size cfg["emb_dim"] - cfg["bottleneck_dim"]
        x_bottleneck = x[:, :, : self.cfg["bottleneck_dim"]]
        x_normal = x[:, :, self.cfg["bottleneck_dim"] :]
        shortcut = x_bottleneck
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back
        # x = torch.cat([x, x_normal], dim=-1)

        return x


class BottleneckFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.fc4 = nn.Linear(cfg["emb_dim"], cfg["bottleneck_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        x_fc3 = self.fc3(x)
        x = nn.functional.silu(x_fc3)
        x_fc4 = self.fc4(x)
        return x_fc4


class BottleneckEncoder(nn.Module):
    def __init__(self, bottleneck_dim: int, emb_dim: int, dtype: torch.dtype = None):
        super().__init__()
        self.bottleneck_encoder = nn.Sequential(
            nn.Linear(emb_dim, bottleneck_dim, bias=False, dtype=dtype),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim, bias=False, dtype=dtype),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.bottleneck_encoder(x)


class BottleneckDecoder(nn.Module):
    def __init__(self, bottleneck_dim: int, emb_dim: int, dtype: torch.dtype = None):
        super().__init__()
        self.bottleneck_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, emb_dim, bias=False, dtype=dtype),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.bottleneck_decoder(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)  # Shape: (b, num_tokens, d_out)
        keys = self.W_key(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)

        # Reshape queries, keys, and values
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (b, num_query_groups, num_tokens, head_dim)

        # Apply RoPE
        keys = apply_rope(keys, cos, sin)
        queries = apply_rope(queries, cos, sin)

        # Expand keys and values to match the number of heads
        # Shape: (b, num_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]

        context_vec = nn.functional.scaled_dot_product_attention(
            query=queries, key=keys, value=values, is_causal=True
        ).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


def compute_rope_params(
    head_dim,
    theta_base=10_000,
    context_length=4096,
    freq_config=None,
    dtype=torch.float32,
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq)

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


def rescale_theta(theta_old, context_length_old, context_length_new):
    scaling_factor = context_length_new / context_length_old
    theta_new = theta_old * scaling_factor
    return theta_new


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
