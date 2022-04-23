import base
import chess
import numpy as np
from typing import List


class RandomEvaluationEngine(base.EvaluationEngine):
    def evaluate_boards(self, boards: List[chess.Board]) -> np.ndarray:
        return np.zeros(len(boards))
