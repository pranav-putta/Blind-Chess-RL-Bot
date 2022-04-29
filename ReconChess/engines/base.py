import chess
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from enum import Enum
from game import Game


class Player(Enum):
    Self = 0
    Opponent = 1

    @staticmethod
    def switch(player):
        if player == Player.Self:
            return Player.Opponent
        else:
            return Player.Self


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
    def random_sample(self) -> Game:
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
