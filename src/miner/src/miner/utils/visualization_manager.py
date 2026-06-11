from __future__ import annotations

import multiprocessing
import threading
import time
import webbrowser

from loguru import logger

from miner.utils.miner_dashboard_api import start_visualization_server


class VisualizationManager:
    def __init__(self, port: int, auto_open: bool = False) -> None:
        self._port = port
        self._auto_open = auto_open
        self._process: multiprocessing.Process | None = None

    def start(self, port: int | None = None) -> None:
        try:
            target_port = port or self._port
            self._process = multiprocessing.Process(
                target=start_visualization_server, args=(target_port,), daemon=True, name="VisualizationServer"
            )
            self._process.start()
            logger.info(f"✅ Visualization server started in separate process (PID: {self._process.pid})")
        except Exception as e:
            logger.exception(f"Error starting visualization server process: {e}")

    def stop(self) -> None:
        if self._process and self._process.is_alive():
            logger.info("Stopping visualization server process...")
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                logger.warning("Visualization server did not terminate gracefully, forcing kill...")
                self._process.kill()
                self._process.join()
            logger.info("✅ Visualization server stopped")

    def open_tab(self, url: str, delay: float = 2.0) -> None:
        def _open() -> None:
            time.sleep(delay)
            try:
                webbrowser.open(url, new=2)
            except Exception as exc:
                logger.warning(f"Could not auto-open visualization tab: {exc}")

        threading.Thread(target=_open, name="VisualizationTabOpener", daemon=True).start()
