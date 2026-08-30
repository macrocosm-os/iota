from pydantic import BaseModel


class RunFlag(BaseModel):
    enabled: bool = False
    version: float = 1.0

    def isOn(self) -> bool:
        return bool(self.enabled)

    def isOff(self) -> bool:
        return not self.isOn()

    def isExactlyOn(self, expected_version: int) -> bool:
        return bool(self.enabled and self.version == expected_version)

    def isAtLeastOn(self, expected_version: int) -> bool:
        return bool(self.enabled and self.version >= expected_version)


class RunFlags(BaseModel):
    test_flag: RunFlag = RunFlag()
    compress_s3_files: RunFlag = RunFlag()
    attest: RunFlag = RunFlag()
    keep_cache_on_local_step: RunFlag = RunFlag()
    upload_optimizer_state: RunFlag = RunFlag()
    enforce_min_local_optimizer_steps: RunFlag = RunFlag()
    weighted_partition_averaging: RunFlag = RunFlag(enabled=True)
    clip_pseudo_gradients: RunFlag = RunFlag()
    use_AdamW: RunFlag = RunFlag()
    peer_eviction: RunFlag = RunFlag()
    auto_max_cache: RunFlag = RunFlag()
    sync_patches: RunFlag = RunFlag(enabled=True)
    auto_local_batch_size: RunFlag = RunFlag(enabled=True)
    # Trusted single-run fleet: disables signature verification, metagraph checks,
    # attestation, and rate limiting on the orchestrator. Never enable for permissionless.
    trusted_fleet: RunFlag = RunFlag()


RUN_FLAGS = RunFlags()


def update_run_flags(new_flags: RunFlags) -> RunFlags:
    """
    Update the shared RUN_FLAGS instance in-place so existing references see the latest values.
    """
    for field_name in new_flags.model_fields:
        new_flag = getattr(new_flags, field_name)
        current_flag = getattr(RUN_FLAGS, field_name, None)

        if isinstance(current_flag, RunFlag):
            current_flag.enabled = new_flag.enabled
            current_flag.version = new_flag.version
        else:
            setattr(RUN_FLAGS, field_name, new_flag)

    return RUN_FLAGS
