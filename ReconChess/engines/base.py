import chess
import numpy as np
from abc import ABC, abstractmethod
from typing import List


class EvaluationEngine(ABC):
    @abstractmethod
    def evaluate_boards(self, boards: List[chess.Board]) -> np.ndarray:
        """
        evaluates a list of chess boards.
        :param boards: list of chess boards
        :return: returns a numpy array of scores
        """
        pass


class SenseEngine(ABC):
    @abstractmethod
    def evaluate_sense(self, boards: List[chess.Board]) -> np.ndarray:
        """
        evaluates all sensing options given a list of chess boards
        :param boards: list of chess boards
        :return: returns a 2-D numpy array of shape (len(boards), 36)
                 with a probability distribution of senses for each board
        """
        pass


class PolicyEngine(ABC):
    @abstractmethod
    def generate_policy(self, state) -> np.ndarray:
        """
        generates a policy for a given state
        :return:
        """
        pass
