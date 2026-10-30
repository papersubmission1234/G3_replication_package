"""Named deterministic random streams for common-random-number experiments."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def stable_uint32_words(*parts: object, n_words: int = 8) -> list[int]:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    words = np.frombuffer(digest, dtype=np.uint32)
    if n_words <= len(words):
        return [int(x) for x in words[:n_words]]
    repeated = np.resize(words, n_words)
    return [int(x) for x in repeated]


@dataclass(frozen=True)
class NamedRNGFactory:
    """Derive reproducible streams from semantic names using SHA-256.
    
    Python's process-randomized hash is unsuitable for seed derivation. Named
    streams here are independent of call order.
    """

    master_seed: int
    namespace: str = "g3-confirmatory-v1"

    def seed_sequence(self, *parts: object) -> np.random.SeedSequence:
        entropy = [int(self.master_seed), *stable_uint32_words(self.namespace, *parts)]
        return np.random.SeedSequence(entropy)

    def generator(self, *parts: object) -> np.random.Generator:
        return np.random.default_rng(self.seed_sequence(*parts))

    def integer_seed(self, *parts: object, high: int = 2_147_483_647) -> int:
        rng = self.generator(*parts)
        return int(rng.integers(0, high, endpoint=False))

    def uniform_table(
        self,
        names: Iterable[object],
        size: int,
        *parts: object,
    ) -> dict[object, np.ndarray]:
        return {
            name: self.generator(*parts, name).random(size=size)
            for name in names
        }
