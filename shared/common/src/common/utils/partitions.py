from typing import Literal
from pydantic import BaseModel
import copy
import random
from loguru import logger


def get_pairs_for_miner(
    miner_hotkeys: list[str], n_partitions: int, target_hotkey: str, seed: int = 42
) -> dict[int, tuple[str, str]]:
    """Assigns cells to pairs of miners. This is used to assign partitions to miners for butterfly all reduce merging.

    Args:
        miner_hotkeys (list[str]): The list of miner hotkeys.
        n_partitions (int): The number of partitions to assign.

    Returns:
        dict[int, tuple[str, str]]: A dictionary of partition numbers to pairs of miner hotkeys.
    """
    # No miners to pair (e.g. no weights submitted this epoch). Returning early
    # avoids the pairing loop below spinning forever on an empty pool.
    if len(miner_hotkeys) == 0:
        logger.warning(f"No miner hotkeys found for target hotkey {target_hotkey}; returning empty list")
        return []

    if len(miner_hotkeys) == 1:
        return {i: (miner_hotkeys[0], None) for i in range(n_partitions)}

    pairs = []
    shuffled_miners = []
    for i in range(n_partitions):
        selected_miners = []
        while True:
            # If we found both miners for the pair, break
            if len(selected_miners) == 2:
                break

            # If we don't have any miners left to choose from, shuffle the list and start over
            if len(shuffled_miners) == 0:
                shuffled_miners = copy.deepcopy(miner_hotkeys)
                random.seed(seed + i)
                random.shuffle(shuffled_miners)

            # If the miner is not already in the pair, add it
            if (selected_miner := shuffled_miners.pop()) not in selected_miners:
                selected_miners.append(selected_miner)

        pairs.append(tuple(selected_miners))

    random.seed(seed + n_partitions)
    random.shuffle(pairs)
    indices = [i for i, pair in enumerate(pairs) if target_hotkey in pair]
    logger.debug(f"Assigning partitions {indices} to miner with hotkey {target_hotkey}")
    random.shuffle(indices)
    return indices


class MinerPartition(BaseModel):
    layer: int | None = None
    chunk_number: int | Literal["all"] = None
    miner_hotkey: str | None = None
    blob_path: str | None = None
    other_miner_hotkey: str | None = None

    def matches(self, other: "MinerPartition") -> bool:
        return (
            self.layer == other.layer
            and self.chunk_number == other.chunk_number
            and self.miner_hotkey == other.miner_hotkey
        )

    def is_valid(self) -> bool:
        return self.blob_path is not None


async def get_start_and_end_indices(tensor_length: int, num_sections: int, target_section: int) -> tuple[int, int]:
    """Get the start and end indices for a tensor.

    Args:
        tensor_length (int): The length of the tensor to get the start and end indices for.
        num_sections (int): The number of sections to split the tensor into.
        target_section (int): The target section to get the start and end indices for.

    Returns:
        tuple[int, int]: The start and end indices for the target section.
    """
    assert target_section < num_sections, "Target section is greater than the number of sections"
    section_size = tensor_length // num_sections
    for i in range(int(min(target_section + 1, num_sections))):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < num_sections - 1 else tensor_length
        assert start_idx is not None and end_idx is not None, "Start idx and end idx are missing"
    return start_idx, end_idx
