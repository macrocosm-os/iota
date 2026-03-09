from common.snowpipe.base import SnowpipeMessage


class TestEvent(SnowpipeMessage):
    """Test message for verifying Snowpipe streaming end-to-end."""

    type: str = "test_event"
    payload: str = ""
