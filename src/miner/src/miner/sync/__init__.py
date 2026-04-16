"""miner.sync: client-side distributed variable synchronisation.

Connect to the bridge sync server and keep local variables in sync across
all miner nodes.

Usage::

    from miner.sync import PollingLoop, SyncedVariable, sync_run_sync_prefix

    # Once at startup, point the shared loop at the bridge:
    SyncedVariable.polling_loop = PollingLoop("http://bridge:8001/sync")

    # Then create variables anywhere — include the run prefix in the id:
    phase = SyncedVariable(f"{sync_run_sync_prefix('default')}/phase", default="idle")
"""

from miner.sync.collections import SyncedDict, SyncedList
from miner.sync.counter import DistributedCounter
from miner.sync.variable import PollingLoop, SyncError, SyncedVariable, sync_run_sync_prefix
from miner.sync.node import SyncedNode
from miner.sync.registry import ComputeNode, NodeRegistry

__all__ = [
    "SyncedDict",
    "SyncedList",
    "DistributedCounter",
    "SyncedVariable",
    "SyncError",
    "PollingLoop",
    "sync_run_sync_prefix",
    "SyncedNode",
    "ComputeNode",
    "NodeRegistry",
]
