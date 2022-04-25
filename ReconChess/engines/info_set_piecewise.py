from abc import ABC, abstractmethod
import chess
from typing import List, Tuple, Any
import ReconChess.engines.base as base


class PiecewiseInformationSet(base.InformationSet):

    def __init__(self, board: base.Game):
        """
        constructs new information set from a board state
        :param board:
        """
        pass

    @property
    def raw(self) -> Any:
        """
        returns raw representation of information set
        :return:
        """
        pass

    def random_sample(self) -> chess.Board:
        """
        randomly generates a sample board state
        :return:
        """
        return None

    def update_with_sense(self, sense_result: List[Tuple[chess.Square, chess.Piece]]):
        pass

    def update_with_move(self, move_result):
        pass

    def propagate_opponent_move(self):
        pass

    def __copy__(self):
        """
        returns a deep copy of the information set
        :return:
        """
        pass
