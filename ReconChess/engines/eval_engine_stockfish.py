import base
import chess
import chess.engine
import numpy as np
from typing import List

STOCKFISH_PATH = '/usr/local/Cellar/stockfish/14.1/bin/stockfish'
EVAL_TIME_LIMIT = 0.1


class StockFishEvaluationEngine(base.EvaluationEngine):
    engine: chess.engine.SimpleEngine

    def __init__(self):
        self.restart_engine()

    def restart_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH, setpgrp=True)

    def evaluate_boards(self, boards: List[chess.Board]) -> np.ndarray:
        arr = []
        for board in boards:
            score = 0
            for attempt in range(1, 5):
                try:
                    score = self.engine.analyse(board, chess.engine.Limit(time=EVAL_TIME_LIMIT))
                    break
                except chess.engine.EngineError:
                    self.restart_engine()
            arr.append(score)
        return np.array(arr)
