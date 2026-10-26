import pytest


@pytest.fixture(autouse=True)
def _isolated_registration_queue_store(tmp_path, monkeypatch):
    # Keep tests from reading or writing the real per-user queue store.
    monkeypatch.setenv("REGISTRATION_QUEUE_STORE_PATH", str(tmp_path / "registration_queue.json"))
