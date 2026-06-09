"""In-process P2P stack for the miner — built on iota-sdk transport.

Replaces the old subprocess-based `common.iroh.p2p_stack.P2PStack`.
Native Rust iroh (via iota-sdk) is robust enough that subprocess
isolation isn't needed: receiver, sender, and the activation cache all
live in the main process, communicating with the rest of the miner via
ordinary Python objects (`asyncio.Queue`, `dict`, `OrderedDict`) instead
of `multiprocessing.Manager` proxies and shared memory.
"""

from miner.p2p.stack import P2PStack, SenderUnavailableError

__all__ = ["P2PStack", "SenderUnavailableError"]
