import json
import os
import time
import uuid
from typing import Any, Dict, Literal, Optional, get_args
from loguru import logger

# API endpoint for the visualization server
VISUALIZATION_API_URL = os.getenv("VISUALIZATION_API_URL", "http://localhost:8009")

counts = {}
_counts_per_hotkey = {}  # Separate counts for each hotkey
_in_memory_events = {}  # In-memory cache of events per hotkey
_in_memory_counts = {}  # In-memory cache of count logs per hotkey

TIMER_NAMES = Literal[
    "download_and_set_global_weights",
    "download_activations",
    "forward",
    "upload_multipart_to_s3",
    "upload_activation",
    "submit_activation",
    "moving to gpu",
    "backward pass",
    "publishing_backwards",
    "cleaning up cache",
    "backward",
    "forward",
    "publish_loss",
    "migrate_activation_state",
    "compute_last_layer_loss",
    "submit_weights",
    "initiate_activation_upload",
    "initiate_blob_upload",
    "upload_multipart_blob_to_s3",
    "merge_partitions",
    "complete_file_upload_request",
]


class TimerLogger:
    _last_count_log_time: float = 0
    _count_log_interval: float = 10.0  # Log every 10 seconds
    _http_client: Any = None

    def __init__(self, name: TIMER_NAMES, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        if self.name not in get_args(TIMER_NAMES):
            raise ValueError(f"Invalid timer name: {name}. Valid names are: {', '.join(get_args(TIMER_NAMES))}")
        self.metadata = metadata or {}
        self.enter_time: Optional[float] = None
        self.exit_time: Optional[float] = None
        self.event_id: str = str(uuid.uuid4())

    async def __aenter__(self):
        self.enter_time = time.time()

        # Write start event
        start_event = {
            "id": self.event_id,
            "name": self.name,
            "type": "start",
            "time": self.enter_time,
            "metadata": self.metadata,
        }
        logger.debug(f"Timer Logger: {json.dumps(start_event)}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exit_time = time.time()
        end_event = {
            "id": self.event_id,
            "name": self.name,
            "type": "end",
            "time": self.exit_time,
            "metadata": self.metadata,
            "duration": self.exit_time - self.enter_time,
        }

        logger.debug(f"Timer Logger: {json.dumps(end_event)}")
        return False
