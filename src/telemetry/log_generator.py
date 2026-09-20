from __future__ import annotations

import numpy as np


class LogGenerator:
    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(abs(seed) + 1)

    def ambient(self, service: str, tick: int) -> list[tuple[str, str]]:
        if tick % 4 == 0:
            return [("INFO", "heartbeat ok")]
        if tick % 31 == 0:
            return [("INFO", "request processed")]
        if tick % 97 == 0:
            return [("WARNING", "slow query detected")]
        if tick % 131 == 0:
            return [("WARNING", "connection retry")]
        if tick % 211 == 0:
            return [("ERROR", "temporary upstream timeout")]
        return []