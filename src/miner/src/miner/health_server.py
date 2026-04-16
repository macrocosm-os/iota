from aiohttp import web
from common import settings as common_settings
from loguru import logger
from miner import settings as miner_settings
from typing import Optional

import subprocess
import time


class HealthServerMixin:
    health_app_runner: Optional[web.AppRunner] = None
    health_site: Optional[web.TCPSite] = None

    def _kill_process_on_port(self, port: int) -> bool:
        """Kill the process using the specified port.

        Returns:
            bool: True if a process was killed, False otherwise
        """
        try:
            # Try to find the process using lsof (works on macOS and Linux)
            result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5)

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                killed_any = False
                for pid in pids:
                    if pid.strip():
                        try:
                            pid_int = int(pid.strip())
                            logger.warning(f"Killing process {pid_int} using port {port}")
                            subprocess.run(["kill", "-9", str(pid_int)], timeout=5, check=False)
                            killed_any = True
                        except (ValueError, subprocess.TimeoutExpired) as e:
                            logger.warning(f"Failed to kill process {pid}: {e}")

                if killed_any:
                    # Give the OS a moment to release the port
                    time.sleep(0.5)
                    return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.debug(f"Could not kill process on port {port} using lsof: {e}")

        return False

    async def _start_health_server(self):
        """Starts the aiohttp web server for healthchecks."""
        app = web.Application()

        async def health_handler(request):
            node_location = getattr(self, "_node_location", None)
            return web.json_response(
                {
                    "status": "healthy",
                    "hotkey": getattr(self, "hotkey", "N/A"),
                    "layer": getattr(self, "state_manager.layer", "N/A"),
                    "uid": getattr(self, "uid", "N/A"),
                    "registered": getattr(self, "registered_on_metagraph", True),
                    "timestamp": time.time(),
                    "spec_version": common_settings.__SPEC_VERSION__,
                    "location": node_location.model_dump() if node_location is not None else None,
                }
            )

        endpoint = getattr(self, "health_endpoint", miner_settings.MINER_HEALTH_ENDPOINT)
        app.router.add_get(endpoint, health_handler)

        if getattr(self, "launch_health", miner_settings.LAUNCH_HEALTH):
            max_retries = 2
            for attempt in range(max_retries):
                # Clean up any existing site/runner before creating new ones
                if self.health_site:
                    try:
                        await self.health_site.stop()
                    except Exception:
                        pass
                    self.health_site = None
                if self.health_app_runner:
                    try:
                        await self.health_app_runner.cleanup()
                    except Exception:
                        pass
                    self.health_app_runner = None

                # Create new runner and site for each attempt
                self.health_app_runner = web.AppRunner(app)
                await self.health_app_runner.setup()

                host = getattr(self, "health_host", miner_settings.MINER_HEALTH_HOST)
                port = getattr(self, "health_port", miner_settings.MINER_HEALTH_PORT)
                endpoint = getattr(self, "health_endpoint", miner_settings.MINER_HEALTH_ENDPOINT)

                self.health_site = web.TCPSite(self.health_app_runner, host, port)

                try:
                    await self.health_site.start()
                    logger.info(
                        f"Miner {getattr(self, 'hotkey', 'N/A')} healthcheck API started on "
                        f"http://{host}:{port}{endpoint}"
                    )
                    break
                except OSError as e:
                    raise

    async def _stop_health_server(self):
        """Stops the aiohttp web server for healthchecks."""
        if self.health_site:
            await self.health_site.stop()
            logger.info(f"Miner {getattr(self, 'hotkey', 'N/A')} healthcheck API site stopped.")
            self.health_site = None
        if self.health_app_runner:
            await self.health_app_runner.cleanup()
            logger.info(f"Miner {getattr(self, 'hotkey', 'N/A')} healthcheck API runner cleaned up.")
            self.health_app_runner = None
