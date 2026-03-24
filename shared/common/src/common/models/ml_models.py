from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_name: str | None = None
    total_global_params: int | None = None
    vocab_size: int | None = None
    context_length: int | None = None
    orig_context_length: int | None = None
    emb_dim: int | None = None
    bottleneck_dim: int | None = None
    quantize_activations: bool | None = None
    quantize_activations_grads: bool | None = None
    quantize_weights: bool | None = None
    n_heads: int | None = None
    n_layers: int | None = None
    hidden_dim: int | None = None
    n_kv_groups: int | None = None
    rope_base: float | None = None
    dtype: str | None = None
    rope_freq: dict[str, float] | None = None
    s3_download_timeout: int | None = None
    s3_upload_timeout: int | None = None
    activation_timeout: int | None = None


class LearningRateMetadata(BaseModel):
    warmup_start_factor: float | None = None
    warmup_steps: int | None = None
    const_steps: int | None = None
    tail_steps_frac: float | None = None
    final_factor: float | None = None
    saw_cycle_length: int | None = None


class OptimizerMetadata(BaseModel):
    learning_rate: float | None = None
    weight_decay: float | None = None
    betas: list[float] | None = None
    eps: float | None = None


class DatasetMetadata(BaseModel):
    dataset_name: str | None = None
    shuffle_dataset: bool | None = None
    sequence_length: int | None = None
    mini_batch_size: int | None = None


class ModelMetadata(BaseModel):
    model_splits: list[list[int]] | None = None
    model_size: str | None = None
    n_splits: int | None = None
    tokenizer_name: str | None = None
    effective_batch_size: int | None = None
    mini_batch_accumulation_count: int | None = None
    grad_clip_norm: float | None = None
    total_train_steps: int | None = None
    lr: LearningRateMetadata | None = None
    opt: OptimizerMetadata | None = None
    dataset: DatasetMetadata | None = None
    miners_per_layer: list[int] | None = None
