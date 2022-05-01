import base as base
import chess
from abc import abstractmethod

from engines import PiecewiseInformationSet


class PiecewiseSenseEngine(base.SenseEngine):
    def choose_sense(self, board: PiecewiseInformationSet) -> chess.Square:
        """
        evaluates all sensing options given a board
        :param board: information set
        :return: returns the chosen square
        """
        return board.piecewisegrid.choose_sense()
