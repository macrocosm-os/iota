import asyncio
import torch
import time
from loguru import logger
from pydantic import BaseModel

from subnet.model import gpu_device
from miner import settings as miner_settings
from common.utils.exceptions import LayerStateException, MinerNotRegisteredException


class ActivationData(BaseModel):
    activation_id: str
    direction: str
    input_activations: torch.Tensor
    sample_activations: torch.Tensor | None
    output_activations: torch.Tensor | None
    state: dict | None
    upload_time: float
    attestation_challenge_blob: str | None = None
    attestation_self_checks: list[str] | None = None
    attestation_crypto: str | None = None

    upload_url: list[str] | None = None
    activation_upload_path: str | None = None

    # P2P push routing: set when activation arrived via push from a peer.
    # Used by the backward pass to know where to send gradients back.
    source_hotkey: str | None = None
    source_p2p_node_ids: list[str] = []

    # URL of the original text sample — passed through push messages so the
    # last-layer miner can download target labels without polling the orchestrator.
    target_download_url: str | None = None

    class Config:
        arbitrary_types_allowed = True


class ActivationCache:
    """
    The ActivationCache is responsible for storing the forward activations that are currently in-process
    so that they are accessible by the backward pass once it is received.
    """

    def __init__(self, hotkey: str, cache_timeout_sec: int):
        self._hotkey: str = hotkey
        self._cache_timeout_sec: float = cache_timeout_sec
        self._cache: dict[str, ActivationData] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

        # Tasks to monitor for exceptions upon reset
        self._removal_tasks: list[asyncio.Task] = []

        # auto_max_cache state: warmup allows unbounded growth; frozen_max_size caps it once set
        self._warmup_active: bool = False
        self._frozen_max_size: int | None = None

    def __len__(self) -> int:
        """Get the number of activations in the cache."""
        return len(self._cache)

    def __contains__(self, activation_id: str) -> bool:
        """Check if an activation_id is in the cache (enables 'in' operator)."""
        return activation_id in self._cache

    def __getitem__(self, activation_id: str) -> ActivationData:
        """Enable cache[activation_id] syntax for getting items."""
        return self._cache[activation_id]

    def __setitem__(self, activation_id: str, activation_data: ActivationData):
        """Enable cache[activation_id] = data syntax for setting items."""
        self._cache[activation_id] = activation_data

    def __delitem__(self, activation_id: str):
        """Enable del cache[activation_id] syntax for deleting items."""
        self._removal_tasks.append(asyncio.create_task(self.remove(activation_id)))

    async def remove(self, activation_id: str):
        """Remove an activation from the cache."""
        async with self._lock:
            logger.debug(f"🗑️ Removing activation {activation_id} from cache")
            try:
                if activation_id not in self._cache:
                    logger.warning(f"Activation {activation_id} has already been removed from cache")
                    return
                activation_data = self._cache[activation_id]
                if activation_data.input_activations is not None:
                    del activation_data.input_activations
                if activation_data.output_activations is not None:
                    del activation_data.output_activations
                del self._cache[activation_id]

                gpu_device.empty_cache()
            except Exception as e:
                logger.error(f"Error removing activation {activation_id} from cache: {e}")
                raise

    @property
    def effective_max_for_queue(self) -> int:
        """Effective max cache size for queue vacancy calculations.

        During auto_max_cache warmup, returns a dynamic ceiling that expands with the cache
        so the queue keeps requesting new forward activations.
        """
        if self._warmup_active:
            return len(self._cache) + miner_settings.MAX_FORWARD_ACTIVATIONS_IN_QUEUE
        return self._frozen_max_size if self._frozen_max_size is not None else miner_settings.MAX_ACTIVATION_CACHE_SIZE

    def is_full(self) -> bool:
        """Check if the cache is full."""
        if self._warmup_active:
            return False
        effective_max = (
            self._frozen_max_size if self._frozen_max_size is not None else miner_settings.MAX_ACTIVATION_CACHE_SIZE
        )
        if len(self._cache) >= effective_max:
            logger.info(
                f"Miner {self._hotkey[:8]} cache full with {len(self._cache)} activations: {self._cache.keys()}"
            )
            self.cleanup()
            return True
        return False

    def cleanup(self):
        """Cleanup the cache of activations that have timed out."""
        for activation_id, activation_data in list(self._cache.items()):
            if activation_data.upload_time < time.time() - self._cache_timeout_sec:
                logger.info(f"🗑️ Removing timed-out activation from cache: {activation_id}")
                del self[activation_id]

    async def reset(self):
        """Reset the cache."""

        # Clear all items in the cache
        activation_ids = list(self._cache.keys())
        for activation_id in activation_ids:
            await self.remove(activation_id)

        # Wait for any removal tasks to complete and raise any exceptions
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*self._removal_tasks, return_exceptions=True),
                timeout=1.0,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error removing activation: {result}")
                    raise result
        except asyncio.TimeoutError:
            logger.warning("Removal tasks timed out")
            pass
        except (LayerStateException, MinerNotRegisteredException) as e:
            # these will have been handled elsewhere
            pass
        except Exception as e:
            logger.error(f"Error during cache reset, waiting for removal tasks to complete: {e}")
            raise
        finally:
            gpu_device.empty_cache()
