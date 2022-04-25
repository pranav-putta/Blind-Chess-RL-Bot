import ReconChess.engines.base as base
import chess
import numpy as np
from typing import List


class RandomEvaluationEngine(base.SimulationEngine):
    def score_boards(self, boards: List[chess.Board]) -> np.ndarray:
        return np.zeros(len(boards))

    def evaluate_rewards(self, boards: List[chess.Board]) -> np.ndarray:
        return np.zeros(len(boards))
