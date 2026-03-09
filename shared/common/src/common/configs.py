LLAMA32_CONFIG_100M = {
    "model_name": "Llama-3.2-100M",
    "total_global_params": 100_000_000,  # This should be manually set for the moment. Attention, model size changes after the first forward pass due to the bottleneck decoder dynamically added to the model.
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 8192,  # Maximum context length to use (reduced to save memory)
    "orig_context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 512,  # Embedding dimension
    "bottleneck_dim": 128,  # Bottleneck dimension, if None
    "quantize_activations": False,
    "quantize_activations_grads": False,
    "quantize_weights": False,
    "n_heads": 32,  # Number of attention heads
    "n_layers": 6,  # Number of layers
    "hidden_dim": 512,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": "torch.bfloat16",  # Lower-precision dtype to reduce memory usage
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "model_splits": [[-1, 3], [3, -1]],
    # "model_splits": [[-1, 2], [2, 4], [4, -1]],
    "miners_per_layer": [1, 1],
    # "miners_per_layer": [1, 1, 1],
    "s3_download_timeout": 180,
    "s3_upload_timeout": 180,
    "activation_timeout": 60,
}

LLAMA32_CONFIG_1B = {
    "model_name": "Llama-3.2-1B",
    "total_global_params": 1_500_055_552,  # This should be manually set for the moment. Attention, model size changes after the first forward pass due to the bottleneck decoder dynamically added to the model.
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 8192,  # Maximum context length to use (reduced to save memory)
    "orig_context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 2048,  # Embedding dimension
    "bottleneck_dim": 32,  # Bottleneck dimension
    "quantize_activations": False,
    "quantize_activations_grads": False,
    "quantize_weights": False,
    "n_heads": 32,  # Number of attention heads
    "n_layers": 16,  # Number of layers
    "hidden_dim": 8192,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": "torch.bfloat16",  # Lower-precision dtype to reduce memory usage
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    # "model_splits": [[-1, 5], [5, 10], [10, -1]],
    "model_splits": [[-1, 4], [4, 12], [12, -1]],  # Even splits
    # "model_splits": [[-1, 2], [2, 6], [6, 10], [10, 14], [14, -1]], # even splits
    "miners_per_layer": [1, 1, 1],
    "s3_download_timeout": 300,
    "s3_upload_timeout": 300,
    "activation_timeout": 60,
}

LLAMA32_CONFIG_3B = {
    "model_name": "Llama-3.2-3B",
    "total_global_params": 3_606_752_256,  # No bottlenecks. Add (2 * embed_dim * bottleneck_dim) for each bottleneck.
    # "total_global_params": 3_606_948_864,  # example. 3 bottlenecks of size 16.
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 8192,  # Maximum context length to use (reduced to save memory)
    "orig_context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 3072,  # Embedding dimension
    "bottleneck_dim": 16,  # Bottleneck dimension, if None
    "quantize_activations": False,
    "quantize_activations_grads": False,
    "quantize_weights": False,
    "n_heads": 24,  # Number of attention heads
    "n_layers": 28,  # Number of layers
    "hidden_dim": 8192,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": "torch.bfloat16",  # Lower-precision dtype to reduce memory usage
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "model_splits": [[-1, 2], [2, 4], [4, -1]],  # TODO: change this to the correct splits
    "miners_per_layer": [1, 1, 2],
    "s3_download_timeout": 600,  # TODO change these to the correct values
    "s3_upload_timeout": 600,
    "activation_timeout": 120,
}

LLAMA32_CONFIG_5B = {
    "model_name": "Llama-3.2-5B",
    "total_global_params": 5_412_917_248,  # No bottlenecks. Add (2 * embed_dim * bottleneck_dim) for each bottleneck.
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 8192,  # Maximum context length to use (reduced to save memory)
    "orig_context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 4096,  # Embedding dimension
    "bottleneck_dim": 16,  # Bottleneck dimension, if None
    "quantize_activations": False,
    "quantize_activations_grads": False,
    "quantize_weights": False,
    "n_heads": 32,  # Number of attention heads
    "n_layers": 20,  # Number of layers
    "hidden_dim": 14336,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": "torch.bfloat16",  # Lower-precision dtype to reduce memory usage
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "model_splits": [[-1, 2], [2, 4], [4, -1]],  # TODO: change this to the correct splits
    "miners_per_layer": [1, 1, 2],
    "s3_download_timeout": 600,  # TODO change these to the correct values
    "s3_upload_timeout": 600,
    "activation_timeout": 120,
}

LLAMA32_CONFIG_7B = {
    "model_name": "Llama-3.2-7B",
    "total_global_params": 7_157_813_248,  # No bottlenecks. Add (2 * embed_dim * bottleneck_dim) for each bottleneck.
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 8192,  # Maximum context length to use (reduced to save memory)
    "orig_context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 4096,  # Embedding dimension
    "bottleneck_dim": 16,  # Bottleneck dimension, if None
    "quantize_activations": False,
    "quantize_activations_grads": False,
    "quantize_weights": False,
    "n_heads": 32,  # Number of attention heads
    "n_layers": 28,  # Number of layers
    "hidden_dim": 14336,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": "torch.bfloat16",  # Lower-precision dtype to reduce memory usage
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "model_splits": [[-1, 2], [2, 4], [4, -1]],  # TODO: change this to the correct splits
    "miners_per_layer": [1, 1, 2],
    "s3_download_timeout": 600,  # TODO change these to the correct values
    "s3_upload_timeout": 600,
    "activation_timeout": 120,
}

LLAMA32_CONFIG_8B = {
    "model_name": "Llama-3.2-8B",
    "total_global_params": 8_030_261_248,  # No bottlenecks. Add (2 * embed_dim * bottleneck_dim) for each bottleneck.
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 8192,  # Maximum context length to use (reduced to save memory)
    "orig_context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 4096,  # Embedding dimension
    "bottleneck_dim": 64,  # Bottleneck dimension, if None
    "quantize_activations": False,
    "quantize_activations_grads": False,
    "quantize_weights": False,
    "n_heads": 32,  # Number of attention heads
    "n_layers": 32,  # Number of layers
    "hidden_dim": 14336,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": "torch.bfloat16",  # Lower-precision dtype to reduce memory usage
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "model_splits": [[-1, 2], [2, 4], [4, -1]],  # TODO: change this to the correct splits
    "miners_per_layer": [1, 1, 2],
    "s3_download_timeout": 600,  # TODO change these to the correct values
    "s3_upload_timeout": 600,
    "activation_timeout": 120,
}

LLAMA32_CONFIG_12B = {
    "model_name": "Llama-3.2-13B",
    "total_global_params": 12_428_661_760,  # No bottlenecks. Add (2 * embed_dim * bottleneck_dim) for each bottleneck.
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 8192,  # Maximum context length to use (reduced to save memory)
    "orig_context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 5120,  # Embedding dimension
    "bottleneck_dim": 16,  # Bottleneck dimension, if None
    "quantize_activations": False,
    "quantize_activations_grads": False,
    "quantize_weights": False,
    "n_heads": 40,  # Number of attention heads
    "n_layers": 40,  # Number of layers
    "hidden_dim": 13824,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": "torch.bfloat16",  # Lower-precision dtype to reduce memory usage
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "model_splits": [[-1, 11], [11, 27], [27, -1]],
    "miners_per_layer": [1, 1, 2],
    "s3_download_timeout": 600,  # TODO change these to the correct values
    "s3_upload_timeout": 600,
    "activation_timeout": 120,
}

LLAMA32_CONFIG_15B = {
    "model_name": "Llama-3.2-15B",
    "total_global_params": 15_076_418_560,  # No bottlenecks. Add (2 * embed_dim * bottleneck_dim) for each bottleneck.
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 8192,  # Maximum context length to use (reduced to save memory)
    "orig_context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 5120,  # Embedding dimension
    "bottleneck_dim": 80,  # Bottleneck dimension, if None
    "quantize_activations": False,
    "quantize_activations_grads": False,
    "quantize_weights": False,
    "n_heads": 40,  # Number of attention heads
    "n_layers": 50,  # Number of layers
    "hidden_dim": 13824,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": "torch.bfloat16",  # Lower-precision dtype to reduce memory usage
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "model_splits": [[-1, 8], [8, 19], [19, 30], [30, 41], [41, -1]],
    "miners_per_layer": [1, 1, 1, 1, 3],
    "s3_download_timeout": 600,
    "s3_upload_timeout": 600,
    "activation_timeout": 120,
}

LLAMA32_CONFIG_21B = {
    "model_name": "Llama-3.2-21B",
    "total_global_params": 21_446_661_120,  # No bottlenecks. Add (2 * embed_dim * bottleneck_dim) for each bottleneck.
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 8192,  # Maximum context length to use (reduced to save memory)
    "orig_context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 5120,  # Embedding dimension
    "bottleneck_dim": 80,  # Bottleneck dimension, if None
    "quantize_activations": False,
    "quantize_activations_grads": False,
    "quantize_weights": False,
    "n_heads": 40,  # Number of attention heads
    "n_layers": 64,  # Number of layers
    "hidden_dim": 16384,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": "torch.bfloat16",  # Lower-precision dtype to reduce memory usage
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "model_splits": [[-1, 11], [11, 27], [27, 43], [43, 59], [59, -1]],
    "miners_per_layer": [1, 1, 2],
    "s3_download_timeout": 600,  # TODO change these to the correct values
    "s3_upload_timeout": 600,
    "activation_timeout": 120,
}
