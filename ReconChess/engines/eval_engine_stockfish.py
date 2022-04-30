import engines.base as base
import chess
import chess.engine
import numpy as np
from typing import List

STOCKFISH_PATH = '/usr/local/Cellar/stockfish/14.1/bin/stockfish'


class StockFishEvaluationEngine(base.SimulationEngine):
    engine: chess.engine.SimpleEngine

    def __init__(self, depth=10, color=chess.WHITE):
        self.depth = depth
        self.color = color
        self.restart_engine()

    def restart_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH, setpgrp=True)

    def close(self):
        self.engine.close()

    def check_king_under_attack(self, board, color):
        enemy_king_square = board.king(not color)
        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = board.attackers(color, enemy_king_square)
            if enemy_king_attackers:
                return 200 if color == self.color else -200

        return None

    def score_boards(self, boards: List[chess.Board]) -> np.ndarray:
        arr = []

        for board in boards:

            opp_king = self.check_king_under_attack(board, self.color)
            self_king = self.check_king_under_attack(board, not self.color)

            if self_king is not None:
                arr.append(self_king)
                continue
            if opp_king is not None:
                arr.append(opp_king)
                continue

            score = 0
            for attempt in range(1, 5):
                try:
                    score = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
                    break
                except chess.engine.EngineError as e:
                    print(e, board.fen())
                    self.restart_engine()
                except Exception as e:
                    print(e, board.fen())
                    self.restart_engine()
            try:
                arr.append(score['score'].relative.cp)
            except AttributeError as e:
                # we're at a mate position
                arr.append(200)
        return np.array(arr)
