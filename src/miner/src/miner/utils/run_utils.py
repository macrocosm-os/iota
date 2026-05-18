from common.models.api_models import RunInfo


# TODO: consolidate these functions into one
def identify_best_run(run_info_list: list[RunInfo], is_miner_pool: bool = False) -> RunInfo:
    # best run is based on
    # 1. authorized = True and is_miner_pool matches
    # 2. prefer runs this miner has a force_allow whitelist entry for (regardless of run.whitelisted)
    # 3. then prefer run-level whitelisted runs this miner is authorized for
    # 4. among those, highest incentive_perc reduced by burn_factor and divided by (num_miners + 1)

    applicable_runs = [r for r in run_info_list if r.authorized and r.is_miner_pool == is_miner_pool]
    if len(applicable_runs) == 0:
        raise Exception("Fatal Error: No applicable runs found")

    if len(applicable_runs) == 1:
        return applicable_runs[0]

    def _score(x: RunInfo) -> float:
        return x.incentive_perc * (1 - x.burn_factor) / (x.num_miners + 1)

    best_run = max(
        applicable_runs,
        key=lambda x: (int(x.whitelist_force_allow), int(x.whitelisted and x.authorized), _score(x)),
    )
    return best_run


def get_miner_pool_run(run_info_list: list[RunInfo]) -> RunInfo:
    return identify_best_run(run_info_list=run_info_list, is_miner_pool=True)
