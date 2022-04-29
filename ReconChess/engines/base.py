import chess
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from enum import Enum


class Player(Enum):
    Self = 0
    Opponent = 1


class Game:
    def __init__(self, board: chess.Board):
        self.board = board

    @staticmethod
    def empty():
        return Game(chess.Board())

    def handle_sense(self, square):
        if square not in list(chess.SQUARES):
            return []

        rank, file = chess.square_rank(square), chess.square_file(square)
        sense_result = []
        for delta_rank in [1, 0, -1]:
            for delta_file in [-1, 0, 1]:
                if 0 <= rank + delta_rank <= 7 and 0 <= file + delta_file <= 7:
                    sense_square = chess.square(file + delta_file, rank + delta_rank)
                    sense_result.append((sense_square, self.board.piece_at(sense_square)))

        return sense_result

    def handle_move(self, move: chess.Move):
        self.board.push(move)
        # TODO: fix this
        return None

    def __copy__(self):
        board = self.board.copy()
        return Game(board)

    def __repr__(self):
        return self.board.__repr__()


class SimulationEngine(ABC):
    @abstractmethod
    def score_boards(self, boards: List[chess.Board]) -> np.ndarray:
        """
        evaluates a list of chess boards.
        :param boards: list of chess boards
        :return: returns a numpy array of scores
        """
        pass

    @abstractmethod
    def evaluate_rewards(self, boards: List[chess.Board]) -> np.ndarray:
        pass


class InformationSet(ABC):

    def __init__(self, board: Game):
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

    @abstractmethod
    def random_sample(self) -> chess.Board:
        """
        randomly generates a sample board state
        :return:
        """
        pass

    @abstractmethod
    def update_with_sense(self, sense_result: List[Tuple[chess.Square, chess.Piece]]):
        pass

    @abstractmethod
    def update_with_move(self, move_result):
        pass

    @abstractmethod
    def propagate_opponent_move(self, possible_moves: List[chess.Move], captured_square: bool,
                                captured_piece: chess.Piece):
        pass

    def __copy__(self):
        """
        returns a deep copy of the information set
        :return:
        """
        pass


class SenseEngine(ABC):
    @abstractmethod
    def choose_sense(self, board: InformationSet) -> chess.Square:
        """
        evaluates all sensing options given a board
        :param board: information set
        :return: returns the chosen square
        """
        pass


class PolicyEngine(ABC):
    def __init__(self, sense_engine: SenseEngine, eval_engine: SimulationEngine):
        self.sense_engine = sense_engine
        self.eval_engine = eval_engine

    @abstractmethod
    def generate_policy(self, self_info_set: InformationSet, other_info_set: InformationSet) -> np.ndarray:
        """
        generates a policy for a given state
        :return:
        """
        pass
