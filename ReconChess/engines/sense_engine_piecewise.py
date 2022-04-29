import engines.base as base
import chess
from abc import abstractmethod


class PiecewiseSenseEngine(base.SenseEngine):
    @abstractmethod
    def choose_sense(self, board: base.InformationSet) -> chess.Square:
        """
        evaluates all sensing options given a board
        :param board: information set
        :return: returns the chosen square
        """
        return 0
