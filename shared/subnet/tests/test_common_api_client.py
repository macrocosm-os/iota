import pytest

from common.utils.exceptions import RateLimitException
from subnet import common_api_client as common_api_client_module
from subnet.common_api_client import CommonAPIClient


class _FailingRequestContext:
    async def __aenter__(self):
        raise TimeoutError("response lost")

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FailingSession:
    def __init__(self, attempts: list[int], **_kwargs):
        self._attempts = attempts

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def request(self, *_args, **_kwargs):
        self._attempts.append(1)
        return _FailingRequestContext()


@pytest.mark.asyncio
async def test_orchestrator_request_honors_single_attempt_override(monkeypatch):
    attempts = []

    monkeypatch.setattr(
        common_api_client_module,
        "ClientSession",
        lambda **kwargs: _FailingSession(attempts, **kwargs),
    )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(common_api_client_module.asyncio, "sleep", no_sleep)

    with pytest.raises(RateLimitException, match="Failed request after 1 attempt"):
        await CommonAPIClient.orchestrator_request(
            method="POST",
            path="/miner/register",
            max_attempts=1,
        )

    assert len(attempts) == 1
