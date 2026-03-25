import json
import time
from miner import settings as miner_settings
from common import settings as common_settings
from typing import Any, Dict, Optional
from common.utils.timer_logger import TimerLogger, TIMER_NAMES
from loguru import logger

try:
    import torch
    import httpx
except ImportError:
    torch = None
import psutil

if not torch:
    logger.warning("Torch is not installed, memory stats will not be available")
else:
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, memory stats will not be available")
    if not torch.backends.mps.is_available():
        logger.warning("MPS is not available, memory stats will not be available")

# API endpoint for the visualization server

counts = {}
_counts_per_hotkey = {}  # Separate counts for each hotkey
_in_memory_events = {}  # In-memory cache of events per hotkey
_in_memory_counts = {}  # In-memory cache of count logs per hotkey


class TimerLoggerMiner(TimerLogger):
    _last_count_log_time: float = 0
    _count_log_interval: float = 10.0  # Log every 10 seconds
    _http_client: Any = None

    def __init__(self, name: TIMER_NAMES, metadata: Optional[Dict[str, Any]] = None, hotkey: Optional[str] = None):
        super().__init__(name=name, metadata=metadata)
        self.hotkey = hotkey[:8] if hotkey else None

        # Initialize in-memory storage for this hotkey if not already done
        if self.hotkey:
            if self.hotkey not in _counts_per_hotkey:
                # _counts_per_hotkey[self.hotkey] = self._load_latest_counts()
                _counts_per_hotkey[self.hotkey] = {}
            if self.hotkey not in _in_memory_events:
                _in_memory_events[self.hotkey] = []
            if self.hotkey not in _in_memory_counts:
                _in_memory_counts[self.hotkey] = []

    @classmethod
    def _get_http_client(cls):
        """Get or create the HTTP client."""
        if cls._http_client is None:
            cls._http_client = httpx.AsyncClient(timeout=2.0)
        return cls._http_client

    def _get_memory_stats(self):
        """Get current memory usage for this process"""
        stats = {}

        # Get memory usage for this process
        try:
            if torch and torch.cuda.is_available():
                vram_allocated = torch.cuda.memory_allocated() / (1024**3)
                vram_reserved = torch.cuda.memory_reserved() / (1024**3)
                stats["vram_allocated_gb"] = vram_allocated
                stats["vram_reserved_gb"] = vram_reserved
            if torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                mps_allocated = torch.mps.current_allocated_memory() / (1024**3)
                stats["mps_allocated_gb"] = mps_allocated
            process = psutil.Process()
            ram_usage = process.memory_info().rss / (1024**3)
            stats["cpu_ram_gb"] = ram_usage
        except Exception as e:
            print(f"✗ Error getting memory stats: {e}")

        return stats

    async def __aenter__(self):
        self.enter_time = time.time()

        # Write start event
        start_event = {
            "id": self.event_id,
            "name": self.name,
            "type": "start",
            "time": self.enter_time,
            "metadata": self.metadata,
            "memory": self._get_memory_stats(),
        }

        # Store in memory
        if self.hotkey and self.hotkey in _in_memory_events:
            _in_memory_events[self.hotkey].append(start_event)

        # Try to send to API (non-blocking)
        logger.debug(f"Sending start event to API: {start_event}")
        await self._send_event_to_api(start_event)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exit_time = time.time()

        # Write end event
        end_event = {
            "id": self.event_id,
            "name": self.name,
            "type": "end",
            "time": self.exit_time,
            "metadata": self.metadata,
            "duration": self.exit_time - self.enter_time,
            "memory": self._get_memory_stats(),
        }

        # Store in memory
        if self.hotkey and self.hotkey in _in_memory_events:
            _in_memory_events[self.hotkey].append(end_event)

        # Try to send to API (non-blocking)
        await self._send_event_to_api(end_event)
        await self._log_event_counts(self.name)
        logger.debug(f"Timer Logger: {json.dumps(end_event)}")
        return False

    async def _send_event_to_api(self, event: Dict[str, Any]):
        """Send an event to the visualization API server."""
        if not self.hotkey or common_settings.NETWORK == "local":
            return

        try:
            client = self._get_http_client()
            await client.post(f"{miner_settings.VISUALIZATION_API_URL}/api/events/{self.hotkey}", json=event)
        except Exception as e:
            # Silently fail - don't block the main process
            logger.debug(f"Failed to send event to visualization API: {e}")

    async def _send_count_log_to_api(self, count_log: Dict[str, Any]):
        """Send a count log to the visualization API server."""
        if not self.hotkey or common_settings.NETWORK == "local":
            return

        try:
            client = self._get_http_client()
            await client.post(f"{miner_settings.VISUALIZATION_API_URL}/api/counts/{self.hotkey}", json=count_log)
        except Exception as e:
            # Silently fail - don't block the main process
            logger.debug(f"Failed to send count log to visualization API: {e}")

    async def _log_event_counts(self, name):
        """Log event counts every 10 seconds - ensures counts are monotonically increasing"""
        # Use per-hotkey counts if available, otherwise fall back to global counts
        if self.hotkey:
            current_counts = _counts_per_hotkey[self.hotkey]
        else:
            current_counts = counts

        # Increment the count for this event
        if name in current_counts:
            current_counts[name] += 1
        else:
            current_counts[name] = 1

        current_time = time.time()

        # Check if 10 seconds have passed since last log
        if current_time - TimerLoggerMiner._last_count_log_time < TimerLoggerMiner._count_log_interval:
            return

        TimerLoggerMiner._last_count_log_time = current_time

        # Create count log entry with a copy of current counts
        count_entry = {
            "timestamp": current_time,
            "total_events": sum(current_counts.values()),
            "event_counts": dict(current_counts),  # Make a copy
            "memory": self._get_memory_stats(),
        }

        # Store in memory
        if self.hotkey and self.hotkey in _in_memory_counts:
            _in_memory_counts[self.hotkey].append(count_entry)

        # Try to send to API (non-blocking)
        await self._send_count_log_to_api(count_entry)
