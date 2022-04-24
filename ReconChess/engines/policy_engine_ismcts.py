import chess

import ReconChess.engines.base as base
import numpy as np
from typing import List, Tuple
from ReconChess.prob_board import PiecewiseGrid


class Node:
    idx: int
    belief: PiecewiseGrid
    truth: chess.Board


class ISMCTSPolicyEngine(base.PolicyEngine):
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
