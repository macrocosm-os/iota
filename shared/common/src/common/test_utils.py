"""Shared test utilities for all tests across orchestrator, scheduler, etc."""

from common.models.api_models import RunInfo
from common.models.run_flags import RunFlags


# Test constants
RUN_1_ID = "run_1"
RUN_2_ID = "run_2"
RUN_3_ID = "run_3_miner_pool"
MINER_HOTKEY = "test_hotkey_123"
MINER_COLDKEY = "test_coldkey_456"


class FakeDBClient:
    class _Transaction:
        def __init__(self, session):
            self._session = session

        def __enter__(self):
            return self._session

        def __exit__(self, exc_type, exc, tb):
            return False

    def __init__(self, session):
        self._session = session

    def transaction(self):
        return self._Transaction(self._session)


def create_test_run_info(
    run_id: str,
    is_miner_pool: bool = False,
    authorized: bool = True,
    is_default: bool = False,
    whitelisted: bool = False,
    num_miners: int = 0,
    burn_factor: float = 0.0,
    incentive_perc: float = 0.33,
    max_miners: int = 256,
    whitelist_force_allow: bool = False,
) -> RunInfo:
    return RunInfo(
        run_id=run_id,
        is_default=is_default,
        num_miners=num_miners,
        is_miner_pool=is_miner_pool,
        whitelisted=whitelisted,
        burn_factor=burn_factor,
        incentive_perc=incentive_perc,
        authorized=authorized,
        run_flags=RunFlags(),
        max_miners=max_miners,
        whitelist_force_allow=whitelist_force_allow,
    )
