from common.models.run_metadata import RunTier
from common.test_utils import create_test_run_info
from common.utils.run_selection import rank_registration_runs


def test_highest_explicit_tier_wins():
    runs = [
        create_test_run_info("iron", tier=RunTier.IRON, incentive_perc=1),
        create_test_run_info("gold", tier=RunTier.GOLD, incentive_perc=0.01),
        create_test_run_info("silver", tier=RunTier.SILVER, incentive_perc=0.5),
    ]

    ranked = rank_registration_runs(runs, is_miner_pool=False)

    assert [run.run_id for run in ranked] == ["gold", "silver", "iron"]


def test_same_tier_preserves_capacity_adjusted_incentive_order():
    runs = [
        create_test_run_info("crowded", tier=RunTier.BRONZE, incentive_perc=1, num_miners=9),
        create_test_run_info("open", tier=RunTier.BRONZE, incentive_perc=0.5, num_miners=0),
    ]

    ranked = rank_registration_runs(runs, is_miner_pool=False)

    assert [run.run_id for run in ranked] == ["open", "crowded"]


def test_force_allow_beats_higher_tier():
    runs = [
        create_test_run_info("gold", tier=RunTier.GOLD),
        create_test_run_info("forced-legacy", whitelist_force_allow=True),
    ]

    ranked = rank_registration_runs(runs, is_miner_pool=False)

    assert ranked[0].run_id == "forced-legacy"


def test_same_tier_prefers_whitelisted_run():
    runs = [
        create_test_run_info("general", tier=RunTier.IRON, incentive_perc=1),
        create_test_run_info("whitelisted", tier=RunTier.IRON, whitelisted=True, incentive_perc=0.01),
    ]

    ranked = rank_registration_runs(runs, is_miner_pool=False)

    assert [run.run_id for run in ranked] == ["whitelisted", "general"]


def test_unauthorized_and_wrong_pool_runs_are_excluded():
    runs = [
        create_test_run_info("unauthorized", tier=RunTier.GOLD, authorized=False),
        create_test_run_info("pool", tier=RunTier.GOLD, is_miner_pool=True),
        create_test_run_info("eligible", tier=RunTier.IRON),
    ]

    ranked = rank_registration_runs(runs, is_miner_pool=False)

    assert [run.run_id for run in ranked] == ["eligible"]
