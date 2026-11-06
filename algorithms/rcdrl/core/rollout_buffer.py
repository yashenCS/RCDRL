import random
from typing import List

from algorithms.rcdrl.core.types import Episode, Transition


class RolloutBuffer:


    def __init__(self):
        self.storage: List[Transition] = []

    def clear(self) -> None:

        self.storage.clear()

    def add_transition(self, transition: Transition) -> None:

        self.storage.append(transition)

    def add_episode(self, episode: Episode) -> None:

        for transition in episode.transitions:
            self.add_transition(transition)

    def sample(self, batch_size: int) -> List[Transition]:

        batch_size = min(int(batch_size), len(self.storage))
        if batch_size <= 0:
            return []
        return random.sample(self.storage, batch_size)

    def __len__(self) -> int:
        return len(self.storage)
