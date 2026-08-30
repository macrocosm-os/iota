"""Chunked cross-entropy must match torch.nn.CrossEntropyLoss exactly
(mean over labels != -100), including pad masking and gradient flow."""

import torch

from subnet.model.utils import _chunked_cross_entropy, compute_loss


def _reference(logits, targets, pad_token_id, pack):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    if not pack:
        pad_mask = shift_labels == pad_token_id
        zeros = torch.zeros_like(shift_labels[..., :1])
        pad_mask = torch.cat((zeros, pad_mask[..., :-1]), dim=-1).bool()
        shift_labels[pad_mask] = -100
    return torch.nn.CrossEntropyLoss()(shift_logits.view(-1, logits.shape[-1]).float(), shift_labels.view(-1))


def test_matches_reference_with_padding_and_grads():
    torch.manual_seed(0)
    B, S, V, PAD = 3, 700, 257, 5  # S spans multiple chunks when _CE_CHUNK_TOKENS is monkeypatched
    import subnet.model.utils as u

    old = u._CE_CHUNK_TOKENS
    u._CE_CHUNK_TOKENS = 128  # force many chunks
    try:
        logits = torch.randn(B, S, V, dtype=torch.float32, requires_grad=True)
        logits_ref = logits.detach().clone().requires_grad_(True)
        targets = torch.randint(0, V, (B, S))
        targets[0, 100:] = PAD  # padded tail
        targets[1, :] = PAD  # fully padded row (only first EOS counts)

        ref = _reference(logits_ref, targets.clone(), PAD, pack=False)
        got = compute_loss(
            mock=False,
            logits=logits,
            targets=targets.clone(),
            vocab_size=V,
            pad_token_id=PAD,
            pack=False,
            device="cpu",
        )
        assert torch.allclose(got, ref, atol=1e-5), (got.item(), ref.item())

        ref.backward()
        got.backward()
        assert torch.allclose(logits.grad, logits_ref.grad, atol=1e-5)
        assert logits.grad.abs().sum() > 0
    finally:
        u._CE_CHUNK_TOKENS = old


def test_matches_reference_packed_bf16():
    torch.manual_seed(1)
    B, S, V = 2, 300, 128
    logits = torch.randn(B, S, V, dtype=torch.bfloat16, requires_grad=True)
    logits_ref = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, V, (B, S))
    ref = _reference(logits_ref, targets.clone(), pad_token_id=0, pack=True)
    got = compute_loss(
        mock=False,
        logits=logits,
        targets=targets.clone(),
        vocab_size=V,
        pad_token_id=0,
        pack=True,
        device="cpu",
    )
    # bf16 forward, fp32 upcast per chunk vs whole-tensor: small tolerance
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3), (got.item(), ref.item())


def test_all_ignored_returns_zero_not_nan():
    logits = torch.randn(1, 4, 16, requires_grad=True)
    labels = torch.full((1, 3), -100)
    out = _chunked_cross_entropy(logits, labels, 16)
    assert out.item() == 0.0
