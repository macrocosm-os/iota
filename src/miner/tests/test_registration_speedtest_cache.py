from miner.utils import utils


def test_run_speedtest_cached_reuses_result_within_ttl(monkeypatch):
    calls = []
    now = 100.0

    def fake_speedtest():
        calls.append(len(calls))
        return {"download_mbps": 100.0 + len(calls), "upload_mbps": 50.0}

    monkeypatch.setattr(utils.time, "monotonic", lambda: now)
    monkeypatch.setattr(utils, "run_speedtest", fake_speedtest)
    monkeypatch.setattr(utils.miner_settings, "REGISTRATION_SPEED_TEST_CACHE_TTL_SEC", 3600)
    utils._speedtest_cache_set = False
    utils._speedtest_cache_expires_at = 0.0
    utils._speedtest_cache_result = None

    first = utils.run_speedtest_cached()
    second = utils.run_speedtest_cached()

    assert first == {"download_mbps": 101.0, "upload_mbps": 50.0}
    assert second == first
    assert calls == [0]


def test_run_speedtest_cached_refreshes_after_ttl(monkeypatch):
    calls = []
    now = 100.0

    def fake_speedtest():
        calls.append(len(calls))
        return {"download_mbps": 100.0 + len(calls), "upload_mbps": 50.0}

    monkeypatch.setattr(utils, "run_speedtest", fake_speedtest)
    monkeypatch.setattr(utils.miner_settings, "REGISTRATION_SPEED_TEST_CACHE_TTL_SEC", 10)
    monkeypatch.setattr(utils.time, "monotonic", lambda: now)
    utils._speedtest_cache_set = False
    utils._speedtest_cache_expires_at = 0.0
    utils._speedtest_cache_result = None

    first = utils.run_speedtest_cached()
    now = 111.0
    second = utils.run_speedtest_cached()

    assert first == {"download_mbps": 101.0, "upload_mbps": 50.0}
    assert second == {"download_mbps": 102.0, "upload_mbps": 50.0}
    assert calls == [0, 1]
