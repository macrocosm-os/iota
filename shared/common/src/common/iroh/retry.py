"""
P2P retry and timeout infrastructure for the iroh wrapper.

Centralises all retry logic so callers (Sender, PooledSender) don't need
hand-rolled loops. Provides:

- **P2PTimeouts** – separate budgets for connection, stream-open, send, receive.
- **P2PRetryPolicy** – max retries, exponential backoff, retryable exception
  types, and connection-invalidation behaviour.
- **P2PRetry** – async executor that applies the policy to an arbitrary
  coroutine factory.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Tuple, Type, TypeVar

from loguru import logger

from common.iroh.timings import P2POperationTimings

try:
    from iroh import IrohError
except ImportError:  # iroh not installed (e.g. CI)
    IrohError = None  # type: ignore[assignment,misc]

T = TypeVar("T")


# ---------------------------------------------------------------------------
# IrohError classification
# ---------------------------------------------------------------------------

# Error messages that indicate a single peer connection died — not a
# node-level problem.  For these we just invalidate the one connection
# instead of tearing down the entire iroh node (which kills *all*
# connections to healthy peers).
_CONNECTION_SCOPED_PATTERNS = (
    "connection lost",
    "closed by peer",
    "timed out",
    "stream closed",
    "reset by peer",
)


def _is_connection_scoped_iroh_error(exc: BaseException) -> bool:
    """Return True when *exc* looks like a single-connection failure."""
    msg = str(exc).lower()
    return any(p in msg for p in _CONNECTION_SCOPED_PATTERNS)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class P2PTimeouts:
    """Separate timeouts (seconds) for each phase of a P2P operation.

    Keeping these independent avoids the problem where a slow send phase
    eats the entire budget originally intended for the receive phase.
    """

    connection: float = 5.0  # QUIC connection establishment (DERP relay + NAT)
    stream_open: float = 5.0  # Opening a uni/bi stream on an existing connection
    send: float = 30.0  # Writing request bytes to the stream
    receive: float = 15.0  # Reading response bytes from the stream


@dataclass(frozen=True)
class P2PRetryPolicy:
    """Configurable retry behaviour for P2P operations.

    ``max_retries=0`` means one attempt total (no retries).
    ``max_retries=2`` means up to three attempts.
    """

    max_retries: int = 2
    base_delay: float = 0.25  # seconds
    max_delay: float = 5.0  # seconds
    backoff_factor: float = 2.0  # exponential multiplier
    retryable_exceptions: Tuple[Type[BaseException], ...] = field(default_factory=lambda: (Exception,))
    invalidate_on_timeout: bool = True
    invalidate_on_error: bool = True

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the backoff delay for *attempt* (0-indexed)."""
        return min(self.base_delay * (self.backoff_factor**attempt), self.max_delay)

    @property
    def total_attempts(self) -> int:
        return self.max_retries + 1


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class P2PRetry:
    """Execute an async operation with retries, backoff and per-phase timeouts.

    Usage::

        retry = P2PRetry(policy, timeouts)
        result = await retry.execute(
            lambda: sender.send_message_bi(...),
            on_invalidate=lambda: sender.invalidate_connection(node_id),
        )
    """

    def __init__(self, policy: P2PRetryPolicy, timeouts: P2PTimeouts) -> None:
        self.policy = policy
        self.timeouts = timeouts

    async def execute(
        self,
        coro_factory: Callable[[], Awaitable[T]],
        *,
        timeout: Optional[float] = None,
        on_invalidate: Optional[Callable[[], Awaitable[Any]]] = None,
        on_node_reset: Optional[Callable[[], Awaitable[Any]]] = None,
        timings: Optional[P2POperationTimings] = None,
    ) -> T:
        """Run *coro_factory()* with retries.

        Parameters
        ----------
        coro_factory:
            A zero-arg callable that returns a fresh awaitable on each
            invocation (coroutines are single-use).
        timeout:
            Optional overall timeout applied via ``asyncio.wait_for``.
            Usually ``None`` because per-phase timeouts are already applied
            inside the coroutine.
        on_invalidate:
            Async callback invoked when the cached connection should be
            discarded (e.g. after a timeout or transport error).
        on_node_reset:
            Async callback invoked when an ``IrohError`` is caught,
            indicating the underlying iroh node may be broken and
            should be shut down and recreated on the next attempt.
        timings:
            Optional mutable timing record.  When provided the executor
            records ``attempt_count``, ``retry_count``,
            ``total_backoff_time`` and ``errors`` into it.

        Raises
        ------
        asyncio.TimeoutError
            After all retries are exhausted on timeout.
        Exception
            The last exception after all retries are exhausted.
        """
        last_exc: BaseException | None = None

        for attempt in range(self.policy.total_attempts):
            if timings is not None:
                timings.attempt_count = attempt + 1

            try:
                if timeout is not None:
                    return await asyncio.wait_for(coro_factory(), timeout=timeout)
                return await coro_factory()

            except asyncio.CancelledError:
                raise  # never swallow cancellations

            except asyncio.TimeoutError as exc:
                last_exc = exc
                if timings is not None:
                    timings.errors.append(f"TimeoutError (attempt {attempt + 1})")
                # Always invalidate the connection on timeout — the cached
                # connection is likely dead and reusing it will just hang again.
                if self.policy.invalidate_on_timeout and on_invalidate:
                    try:
                        await on_invalidate()
                    except Exception:
                        pass
                if attempt < self.policy.max_retries:
                    delay = self.policy.delay_for_attempt(attempt)
                    if timings is not None:
                        timings.retry_count = attempt + 1
                        timings.total_backoff_time += delay
                    logger.warning(
                        f"P2P timeout retry {attempt + 1}/{self.policy.total_attempts} "
                        f"— invalidated connection, backoff {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

            except self.policy.retryable_exceptions as exc:
                last_exc = exc
                if timings is not None:
                    timings.errors.append(f"{type(exc).__name__}: {exc} (attempt {attempt + 1})")

                # Classify IrohErrors: connection-scoped errors just need
                # the one connection invalidated; node-scoped errors need a
                # full node reset.
                is_iroh_error = IrohError is not None and isinstance(exc, IrohError)
                conn_scoped = is_iroh_error and _is_connection_scoped_iroh_error(exc)

                if is_iroh_error:
                    if conn_scoped:
                        logger.warning(f"Connection-scoped IrohError, invalidating connection: {exc}")
                        if on_invalidate:
                            try:
                                await on_invalidate()
                            except Exception:
                                pass
                    elif on_node_reset:
                        logger.warning(f"Node-scoped IrohError, resetting node: {exc}")
                        try:
                            await on_node_reset()
                        except Exception:
                            pass

                if attempt < self.policy.max_retries:
                    delay = self.policy.delay_for_attempt(attempt)
                    if timings is not None:
                        timings.retry_count = attempt + 1
                        timings.total_backoff_time += delay
                    logger.warning(
                        f"P2P retry {attempt + 1}/{self.policy.total_attempts}: "
                        f"{type(exc).__name__}: {exc} — backoff {delay:.2f}s"
                    )
                    if self.policy.invalidate_on_error and on_invalidate and not is_iroh_error:
                        try:
                            await on_invalidate()
                        except Exception:
                            pass
                    await asyncio.sleep(delay)
                else:
                    if self.policy.invalidate_on_error and on_invalidate and not is_iroh_error:
                        try:
                            await on_invalidate()
                        except Exception:
                            pass
                    raise

        # Should be unreachable, but satisfies the type checker.
        assert last_exc is not None
        raise last_exc
