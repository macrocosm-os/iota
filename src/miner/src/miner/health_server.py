from aiohttp import web
from common import settings as common_settings
from loguru import logger
from miner import settings as miner_settings
from typing import Optional

import time


class HealthServerMixin:
    health_app_runner: Optional[web.AppRunner] = None
    health_site: Optional[web.TCPSite] = None

    async def _start_health_server(self):
        """Starts the aiohttp web server for healthchecks."""
        app = web.Application()

        async def health_handler(request):
            return web.json_response(
                {
                    "status": "healthy",
                    "hotkey": getattr(self, "hotkey", "N/A"),
                    "run_id": getattr(self.state_manager, "run_id", "N/A"),
                    "layer": getattr(self.state_manager, "layer", "N/A"),
                    "backwards_since_reset": getattr(self.training_phase, "backwards_since_reset", "N/A"),
                    "training_epoch_when_registered": getattr(self.state_manager, "training_epoch_when_registered", 0),
                    "timestamp": time.time(),
                    "spec_version": common_settings.__SPEC_VERSION__,
                }
            )

        app.router.add_get(miner_settings.MINER_HEALTH_ENDPOINT, health_handler)

        self.health_app_runner = web.AppRunner(app)
        await self.health_app_runner.setup()

        self.health_site = web.TCPSite(
            self.health_app_runner, miner_settings.MINER_HEALTH_HOST, miner_settings.MINER_HEALTH_PORT
        )
        if miner_settings.LAUNCH_HEALTH:
            await self.health_site.start()
            logger.info(
                f"Miner {getattr(self, 'hotkey', 'N/A')} healthcheck API started on "
                f"http://{miner_settings.MINER_HEALTH_HOST}:{miner_settings.MINER_HEALTH_PORT}{miner_settings.MINER_HEALTH_ENDPOINT}"
            )

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
