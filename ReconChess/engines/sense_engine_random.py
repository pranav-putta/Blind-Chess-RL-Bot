import engines.base as base
import chess
import numpy as np
from typing import List


class RandomSenseEngine(base.SenseEngine):
    def choose_sense(self, board: base.InformationSet) -> chess.Square:
        """
        Returns a uniform distribution to sense
        :param boards:
        :return:
        """
        dist = np.ones(36) / 36
        return chess.A1
