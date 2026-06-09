"""Miner host hardware + runtime resource gauges.

Lives on its own ``HARDWARE_REGISTRY`` so static specs and live resource
usage stay separate from training-loop metrics (which belong on
``metric_registry.MINER_REGISTRY``). Both registries are snapshotted on
each fleet-telemetry flush and ride the same prometheus_snapshots
payload — the split is purely about ownership, not transport.
"""

from __future__ import annotations

import os

import psutil
import torch
from loguru import logger
from prometheus_client import CollectorRegistry, Gauge

from miner.utils.utils import collect_hardware_info

PREFIX = "miner_"

HARDWARE_REGISTRY = CollectorRegistry()

# --- Static (info pattern, set once at startup) ----------------------------
HW_INFO = Gauge(
    f"{PREFIX}hw_info",
    "Constant=1 info metric carrying static host descriptors as labels.",
    labelnames=["machine", "cpu_brand", "gpu_name"],
    registry=HARDWARE_REGISTRY,
)
HW_GPU_COUNT = Gauge(
    f"{PREFIX}hw_gpu_count",
    "Number of CUDA devices visible to the miner process.",
    labelnames=["gpu_name"],
    registry=HARDWARE_REGISTRY,
)
HW_MEMORY_TOTAL_BYTES = Gauge(
    f"{PREFIX}hw_memory_total_bytes",
    "Total system RAM in bytes (static).",
    registry=HARDWARE_REGISTRY,
)
HW_CPU_CORES = Gauge(
    f"{PREFIX}hw_cpu_cores",
    "Logical CPU core count (static).",
    registry=HARDWARE_REGISTRY,
)

# --- Live VRAM (per CUDA device, sampled per flush) ------------------------
VRAM_USED_BYTES = Gauge(
    f"{PREFIX}vram_used_bytes",
    "VRAM bytes in use on the device (driver view: total - free).",
    labelnames=["gpu_index", "gpu_name"],
    registry=HARDWARE_REGISTRY,
)
VRAM_TOTAL_BYTES = Gauge(
    f"{PREFIX}vram_total_bytes",
    "Total VRAM bytes on the device.",
    labelnames=["gpu_index", "gpu_name"],
    registry=HARDWARE_REGISTRY,
)
VRAM_ALLOCATED_BYTES = Gauge(
    f"{PREFIX}vram_allocated_bytes",
    "Bytes torch has allocated for live tensors on the device.",
    labelnames=["gpu_index", "gpu_name"],
    registry=HARDWARE_REGISTRY,
)
VRAM_RESERVED_BYTES = Gauge(
    f"{PREFIX}vram_reserved_bytes",
    "Bytes torch holds in its caching allocator on the device.",
    labelnames=["gpu_index", "gpu_name"],
    registry=HARDWARE_REGISTRY,
)

# --- Live host RAM / disk / CPU --------------------------------------------
RAM_USED_BYTES = Gauge(
    f"{PREFIX}ram_used_bytes",
    "Host RAM bytes in use (psutil.virtual_memory().used).",
    registry=HARDWARE_REGISTRY,
)
RAM_AVAILABLE_BYTES = Gauge(
    f"{PREFIX}ram_available_bytes",
    "Host RAM bytes available for new allocations.",
    registry=HARDWARE_REGISTRY,
)
RAM_TOTAL_BYTES = Gauge(
    f"{PREFIX}ram_total_bytes",
    "Total host RAM bytes (psutil.virtual_memory().total).",
    registry=HARDWARE_REGISTRY,
)
DISK_USED_BYTES = Gauge(
    f"{PREFIX}disk_used_bytes",
    "Disk bytes used on the given mount.",
    labelnames=["path"],
    registry=HARDWARE_REGISTRY,
)
DISK_TOTAL_BYTES = Gauge(
    f"{PREFIX}disk_total_bytes",
    "Total disk bytes on the given mount.",
    labelnames=["path"],
    registry=HARDWARE_REGISTRY,
)
CPU_PERCENT = Gauge(
    f"{PREFIX}cpu_percent",
    "Host CPU utilization percent since the last sample (non-blocking).",
    registry=HARDWARE_REGISTRY,
)
PROCESS_RSS_BYTES = Gauge(
    f"{PREFIX}process_rss_bytes",
    "Resident set size of the miner process in bytes.",
    registry=HARDWARE_REGISTRY,
)
PROCESS_CPU_PERCENT = Gauge(
    f"{PREFIX}process_cpu_percent",
    "Miner process CPU percent since the last sample.",
    registry=HARDWARE_REGISTRY,
)

# --- Optional NVML gauges (only registered if pynvml importable) ----------
try:
    import pynvml  # type: ignore[import-untyped]

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception as _nvml_err:
    pynvml = None  # type: ignore[assignment]
    _NVML_AVAILABLE = False
    logger.debug(f"pynvml unavailable, NVML gauges disabled: {_nvml_err}")

if _NVML_AVAILABLE:
    GPU_UTILIZATION_PERCENT = Gauge(
        f"{PREFIX}gpu_utilization_percent",
        "GPU compute utilization percent (NVML).",
        labelnames=["gpu_index", "gpu_name"],
        registry=HARDWARE_REGISTRY,
    )
    GPU_TEMPERATURE_CELSIUS = Gauge(
        f"{PREFIX}gpu_temperature_celsius",
        "GPU core temperature in Celsius (NVML).",
        labelnames=["gpu_index", "gpu_name"],
        registry=HARDWARE_REGISTRY,
    )
    GPU_POWER_WATTS = Gauge(
        f"{PREFIX}gpu_power_watts",
        "GPU instantaneous power draw in watts (NVML).",
        labelnames=["gpu_index", "gpu_name"],
        registry=HARDWARE_REGISTRY,
    )


def _safe_gpu_name(index: int) -> str:
    try:
        return torch.cuda.get_device_name(index) or "unknown"
    except Exception:
        return "unknown"


def record_hardware_info() -> None:
    """Populate the static HW_* gauges. Idempotent; safe to call repeatedly.

    Pulled once at miner startup. Never raises — a probe failure produces
    a debug log and missing labels, not a service crash.
    """
    try:
        hw = collect_hardware_info()
        machine = hw.get("machine") or "unknown"
        cpu_brand = hw.get("cpu_brand") or "unknown"
        gpu_name = hw.get("gpu_name") or "none"

        HW_INFO.labels(machine=machine, cpu_brand=cpu_brand, gpu_name=gpu_name).set(1)
        HW_GPU_COUNT.labels(gpu_name=gpu_name).set(hw.get("gpu_count") or 0)
        if hw.get("memory_bytes"):
            HW_MEMORY_TOTAL_BYTES.set(hw["memory_bytes"])
        if hw.get("cpu_cores"):
            HW_CPU_CORES.set(hw["cpu_cores"])
    except Exception as e:
        logger.warning(f"record_hardware_info failed: {e}")


def _sample_vram() -> None:
    if not torch.cuda.is_available():
        return
    try:
        count = torch.cuda.device_count()
    except Exception:
        return
    for i in range(count):
        name = _safe_gpu_name(i)
        labels = {"gpu_index": str(i), "gpu_name": name}
        try:
            free, total = torch.cuda.mem_get_info(i)
            VRAM_USED_BYTES.labels(**labels).set(total - free)
            VRAM_TOTAL_BYTES.labels(**labels).set(total)
        except Exception as e:
            logger.debug(f"vram driver probe gpu={i} failed: {e}")
        try:
            VRAM_ALLOCATED_BYTES.labels(**labels).set(torch.cuda.memory_allocated(i))
            VRAM_RESERVED_BYTES.labels(**labels).set(torch.cuda.memory_reserved(i))
        except Exception as e:
            logger.debug(f"vram torch probe gpu={i} failed: {e}")


def _sample_ram() -> None:
    try:
        vm = psutil.virtual_memory()
        RAM_USED_BYTES.set(vm.used)
        RAM_AVAILABLE_BYTES.set(vm.available)
        RAM_TOTAL_BYTES.set(vm.total)
    except Exception as e:
        logger.debug(f"ram probe failed: {e}")


def _sample_disk(paths: list[str]) -> None:
    for path in paths:
        try:
            usage = psutil.disk_usage(path)
            DISK_USED_BYTES.labels(path=path).set(usage.used)
            DISK_TOTAL_BYTES.labels(path=path).set(usage.total)
        except Exception as e:
            logger.debug(f"disk probe path={path} failed: {e}")


def _sample_cpu_and_process() -> None:
    try:
        # interval=None returns the percent since the previous call without
        # blocking. First call returns 0.0; subsequent samples are meaningful,
        # which is fine on the 15s flush cadence.
        CPU_PERCENT.set(psutil.cpu_percent(interval=None))
    except Exception as e:
        logger.debug(f"cpu_percent probe failed: {e}")
    try:
        proc = psutil.Process()
        PROCESS_RSS_BYTES.set(proc.memory_info().rss)
        PROCESS_CPU_PERCENT.set(proc.cpu_percent(interval=None))
    except Exception as e:
        logger.debug(f"process probe failed: {e}")


def _sample_nvml() -> None:
    if not _NVML_AVAILABLE:
        return
    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception as e:
        logger.debug(f"nvml device count failed: {e}")
        return
    for i in range(count):
        name = _safe_gpu_name(i)
        labels = {"gpu_index": str(i), "gpu_name": name}
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        except Exception as e:
            logger.debug(f"nvml handle gpu={i} failed: {e}")
            continue
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            GPU_UTILIZATION_PERCENT.labels(**labels).set(util.gpu)
        except Exception as e:
            logger.debug(f"nvml util gpu={i} failed: {e}")
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            GPU_TEMPERATURE_CELSIUS.labels(**labels).set(temp)
        except Exception as e:
            logger.debug(f"nvml temp gpu={i} failed: {e}")
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            GPU_POWER_WATTS.labels(**labels).set(power_mw / 1000.0)
        except Exception as e:
            logger.debug(f"nvml power gpu={i} failed: {e}")


def sample_resource_usage(disk_paths: list[str] | None = None) -> None:
    """Update all live resource gauges. Never raises."""
    paths = disk_paths if disk_paths else ["/"]
    _sample_vram()
    _sample_ram()
    _sample_disk(paths)
    _sample_cpu_and_process()
    _sample_nvml()


def disk_paths_from_env(default: tuple[str, ...] = ("/",)) -> list[str]:
    """Read MINER_DISK_PATHS (comma-separated). Falls back to ``default``."""
    raw = os.getenv("MINER_DISK_PATHS", "").strip()
    if not raw:
        return list(default)
    return [p.strip() for p in raw.split(",") if p.strip()]
