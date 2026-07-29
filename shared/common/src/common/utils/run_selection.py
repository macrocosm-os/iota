from common.models.api_models import RunInfo


def rank_registration_runs(run_info_list: list[RunInfo], *, is_miner_pool: bool) -> list[RunInfo]:
    """Rank authorized, pool-compatible runs for registration."""
    applicable_runs = [run for run in run_info_list if run.authorized and run.is_miner_pool == is_miner_pool]

    def score(run: RunInfo) -> float:
        return run.incentive_perc * (1 - run.burn_factor) / (run.num_miners + 1)

    return sorted(
        applicable_runs,
        key=lambda run: (
            int(run.whitelist_force_allow),
            run.tier,
            int(run.whitelisted),
            score(run),
        ),
        reverse=True,
    )
