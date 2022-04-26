from abc import ABC, abstractmethod
import chess
from typing import List, Tuple, Any
import ReconChess.engines.base as base

from prob_board import PiecewiseGrid

import numpy as np

class PiecewiseInformationSet(base.InformationSet):

    def __init__(self, board: base.Game):
        """
        constructs new information set from a board state
        :param board:
        """
        super().__init__(board)
        self.piecewisegrid = PiecewiseGrid(board)

    @property
    def raw(self) -> Any:
        """
        returns raw representation of information set
        :return:
        """
        return self.piecewisegrid.piece_grids

    def random_sample(self) -> chess.Board:
        """
        randomly generates a sample board state
        :return:
        """
        return self.piecewisegrid.gen_board()

    def update_with_sense(self, sense_result: List[Tuple[chess.Square, chess.Piece]]):
        self.piecewisegrid.handle_sense_result(sense_result)

    def update_with_move(self, move_result):
        pass # not implemented yet

    def propagate_opponent_move(self, possible_moves: List[chess.Move], captured_square: bool, captured_piece: chess.Piece):
        self.piecewisegrid.handle_enemy_move(possible_moves, captured_square, captured_piece)

    def __copy__(self):
        new_grid = np.copy(self.piecewisegrid)
        new_infoset = PiecewiseInformationSet(chess.Board())
        new_infoset.piecewisegrid = new_grid
        return new_infoset
