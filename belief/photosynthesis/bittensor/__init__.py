"""Bittensor biasing: SWE-Bench mirror, subnet watcher, task biaser.

Economic honesty: SN62 is winner-take-all and realistic revenue is $0
unless the engine reaches SOTA. The value of this module is the
quality of the *practice distribution*, not TAO income.

All bittensor SDK interactions are strictly read-only. No wallet
operations. No burn transactions. If the SDK isn't installed or the
mainnet is flaky, the watcher fails silently — the daemon keeps
running.
"""

from belief.photosynthesis.bittensor.swebench_mirror import (
    BittensorTask,
    SwebenchMirror,
)
from belief.photosynthesis.bittensor.task_biaser import (
    BITTENSOR_CENTROID_ID,
    TaskBiaser,
)

__all__ = [
    "BITTENSOR_CENTROID_ID",
    "BittensorTask",
    "SwebenchMirror",
    "TaskBiaser",
]
