import ReconChess.engines.base as base
import chess
import numpy as np
from typing import List


class PiecewiseSenseEngine(base.SenseEngine):
    def evaluate_sense(self, boards: List[chess.Board]) -> np.ndarray:
        """
        Returns a uniform distribution to sense
        :param boards:
        :return:
        """
        dist = np.ones(36) / 36
        return np.tile(dist, len(boards))
