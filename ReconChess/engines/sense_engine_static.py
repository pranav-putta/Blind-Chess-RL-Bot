import random

import engines.base as base
import chess
import numpy as np
from typing import List


class StaticSenseEngine(base.SenseEngine):
    def __init__(self, loc):
        self.loc = loc

    def choose_sense(self, board: base.InformationSet) -> chess.Square:
        """
        Returns a uniform distribution to sense
        :param boards:
        :return:
        """
        return self.loc
