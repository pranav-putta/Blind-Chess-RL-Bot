import chess

import base
import numpy as np
from typing import List, Tuple


class Node:
    idx: int
    board: chess.Board


class VanillaMCTSPolicyEngine(base.PolicyEngine):
    def __init__(self):
        initial_shape = (100, 64 * (7 * 8 + 8 + 9))
        self.Q = np.zeros(initial_shape)
        self.N = np.zeros(initial_shape)
        self.P = np.zeros(initial_shape)
        self.visited: List[Node] = []

    def generate_policy(self, state) -> np.ndarray:
        return self.mcts(state)

    def ucb(self):
        return 0

    def mcts(self, state) -> np.ndarray:
        """
        # create chess game with real state and both players' bitboards

        :return:
        """
