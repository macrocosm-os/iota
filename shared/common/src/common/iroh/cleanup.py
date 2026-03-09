"""Synchronous FFI cleanup for Iroh nodes.

When an Iroh node becomes unhealthy, async shutdown may hang because the
node's internal polling machinery is stuck.  The functions here call the
underlying UniFFI free functions directly — these are *synchronous* Rust
``Drop`` calls that release sockets and locks without going through the
async runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from iroh import Iroh


def _force_free_iroh_node(iroh_obj: Iroh) -> None:
    """Synchronously free an Iroh node's Rust resources via FFI.

    This is intended to be called from a thread-pool executor so that the
    (potentially slow) Rust ``Drop`` does not block the event loop.
    """
    from iroh.iroh_ffi import _uniffi_rust_call, _UniffiLib  # type: ignore[attr-defined]

    # Free the inner Node first
    try:
        inner_node = iroh_obj.node()  # returns Node with cloned pointer
        pointer = getattr(inner_node, "_pointer", None)
        if pointer is not None:
            _uniffi_rust_call(_UniffiLib.uniffi_iroh_ffi_fn_free_node, pointer)
            inner_node._pointer = None  # prevent double-free in __del__
    except Exception as e:
        logger.warning(f"force_destroy: failed to free Node: {e}")

    # Free the Iroh wrapper
    try:
        pointer = getattr(iroh_obj, "_pointer", None)
        if pointer is not None:
            _uniffi_rust_call(_UniffiLib.uniffi_iroh_ffi_fn_free_iroh, pointer)
            iroh_obj._pointer = None  # prevent double-free in __del__
    except Exception as e:
        logger.warning(f"force_destroy: failed to free Iroh: {e}")
