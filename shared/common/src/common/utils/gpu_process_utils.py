import glob
import os
import subprocess
import tempfile
from multiprocessing.shared_memory import SharedMemory

from loguru import logger

# Manifest directory for tracking shared memory names across crashes.
# On macOS, /dev/shm/ does not exist, so we need an alternative way to
# discover orphaned segments on startup.
_SHM_MANIFEST_DIR = os.path.join(tempfile.gettempdir(), "iota_shm_manifests")


def _pid_is_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Process exists but we can't signal it
    except OSError:
        return False


def write_shm_manifest(shm_names: list[str]) -> None:
    """Write active shared memory names to a PID-stamped manifest file."""
    os.makedirs(_SHM_MANIFEST_DIR, exist_ok=True)
    path = os.path.join(_SHM_MANIFEST_DIR, f"{os.getpid()}.manifest")
    try:
        with open(path, "w") as f:
            f.write("\n".join(shm_names))
    except Exception as exc:
        logger.warning(f"Failed to write shm manifest: {exc}")


def remove_shm_manifest() -> None:
    """Remove the manifest file for the current process."""
    path = os.path.join(_SHM_MANIFEST_DIR, f"{os.getpid()}.manifest")
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning(f"Failed to remove shm manifest: {exc}")


def kill_stale_gpu_processes():
    """Kill any orphaned processes holding GPU memory from a previous run.

    When a miner crashes or is force-killed, child processes may retain
    GPU memory.  Running this before CUDA init ensures a clean slate.
    """
    my_pid = str(os.getpid())
    try:
        nvidia_devices = glob.glob("/dev/nvidia*")
        if not nvidia_devices:
            return
        result = subprocess.run(
            ["fuser"] + nvidia_devices,
            capture_output=True,
            text=True,
            timeout=5,
        )
        # fuser prints PIDs to stderr
        pids = set(result.stderr.split()) | set(result.stdout.split())
        pids.discard("")
        for pid in pids:
            pid = pid.rstrip("m")  # fuser appends access mode letters
            if pid == my_pid:
                continue
            try:
                # Check if it's a python process before killing
                cmdline = open(f"/proc/{pid}/cmdline").read()
                if "python" not in cmdline.lower():
                    continue
                logger.warning(f"Killing stale GPU process {pid}")
                os.kill(int(pid), 9)
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
                pass
    except FileNotFoundError:
        pass  # fuser not available or no nvidia devices
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        logger.warning(f"GPU cleanup check failed: {e}")


def cleanup_stale_shared_memory():
    """Unlink any orphaned iota_* shared memory segments from a previous run.

    When a miner is killed before cleanup, SharedMemory blocks with names
    like ``iota_<hex>`` persist until explicitly unlinked.

    Two discovery mechanisms are used:
      1. Linux: scan ``/dev/shm/iota_*`` directly.
      2. All platforms (especially macOS where /dev/shm/ does not exist):
         read manifest files written by previous runs and clean up segments
         belonging to dead processes.
    """
    cleaned = 0

    # --- Linux: scan /dev/shm directly ---
    try:
        for path in glob.glob("/dev/shm/iota_*"):
            name = os.path.basename(path)
            try:
                shm = SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
                cleaned += 1
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(f"Failed to unlink shared memory {name}: {exc}")
    except Exception as e:
        logger.warning(f"Shared memory cleanup failed: {e}")

    # --- All platforms: scan manifest files from dead processes ---
    try:
        if os.path.isdir(_SHM_MANIFEST_DIR):
            for manifest_file in glob.glob(os.path.join(_SHM_MANIFEST_DIR, "*.manifest")):
                try:
                    basename = os.path.basename(manifest_file)
                    pid = int(basename.replace(".manifest", ""))
                except ValueError:
                    continue

                if _pid_is_alive(pid):
                    continue  # Process still running, skip

                # Process is dead — clean up its shared memory
                try:
                    with open(manifest_file) as f:
                        names = [line.strip() for line in f if line.strip()]
                except Exception:
                    names = []

                for name in names:
                    try:
                        shm = SharedMemory(name=name, create=False)
                        shm.close()
                        shm.unlink()
                        cleaned += 1
                    except FileNotFoundError:
                        pass
                    except Exception as exc:
                        logger.warning(f"Failed to unlink shared memory {name}: {exc}")

                # Remove the stale manifest
                try:
                    os.remove(manifest_file)
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Manifest-based shared memory cleanup failed: {e}")

    if cleaned:
        logger.info(f"Cleaned up {cleaned} stale shared memory segments")
