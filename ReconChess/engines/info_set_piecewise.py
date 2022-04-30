from abc import ABC, abstractmethod
import chess
from typing import List, Tuple, Any
import engines.base as base

from prob_board import PiecewiseGrid
from engines.game import Game

import numpy as np


class PiecewiseInformationSet(base.InformationSet):

    def __init__(self, board: Game):
        """
        constructs new information set from a board state
        :param board:
        """
        super().__init__(board)
        self.piecewisegrid = PiecewiseGrid(board.truth_board)

    @property
    def raw(self) -> Any:
        """
        returns raw representation of information set
        :return:
        """
        return self.piecewisegrid.piece_grids

    def random_sample(self) -> Game:
        """
        randomly generates a sample board state
        :return:
        """
        board = self.piecewisegrid.gen_board()
        g = Game()
        g.truth_board = board
        g.white_board = board
        g.black_board = board
        return g

    def size(self):
        entropy = 0
        for i in range(32):
            probs = self.piecewisegrid.piece_grids[:, :, i].flatten()
            entropy += np.sum(-probs * np.log2(probs + 1e-10))
        return round(2 ** entropy)

    def update_with_sense(self, sense_result: List[Tuple[chess.Square, chess.Piece]]):
        self.piecewisegrid.handle_sense_result(sense_result)

    def update_with_move(self, move_result):
        move, captured_piece = move_result
        self.piecewisegrid.handle_player_move(move, captured_piece)

    def propagate_opponent_move(self, possible_moves: List[chess.Move], captured_square: bool,
                                captured_piece: chess.Piece):
        self.piecewisegrid.handle_enemy_move(possible_moves, captured_square, captured_piece)

    def mirror(self):
        return self.piecewisegrid.mirror()

    def __copy__(self):
        new_grid = self.piecewisegrid.__copy__()
        new_infoset = PiecewiseInformationSet(Game())
        new_infoset.piecewisegrid = new_grid
        return new_infoset

    def __repr__(self):
        sample = self.random_sample()
        return str(sample.truth_board)

    def __str__(self):
        sample = self.random_sample()
        return str(sample.truth_board)
