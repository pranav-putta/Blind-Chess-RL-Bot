import engines.base as base
import chess
import chess.engine
import numpy as np
from typing import List
import util

STOCKFISH_PATH = '/usr/local/Cellar/stockfish/14.1/bin/stockfish'
EVAL_TIME_LIMIT = 0.05

class StockFishEvaluationEngine(base.SimulationEngine):
    engine: chess.engine.SimpleEngine

    def __init__(self):
        self.restart_engine()

    def restart_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH, setpgrp=True)

    def score_boards(self, boards: List[chess.Board]) -> np.ndarray:
        arr = []
        for board in boards:
            score = 0
            #for attempt in range(1, 5):
            try:
                score = self.engine.analyse(board, chess.engine.Limit(time=EVAL_TIME_LIMIT))['score'].relative.cp
                #score = self.engine.analyse(board, chess.engine.Limit(depth=20))['score'].relative.cp
                break
            except AttributeError as ae:
                score = 100000
                print("Mate found")
            except chess.engine.EngineError:
                print("ENGINE ERROR WHEN ATTEMPTING TO SCORE BOARD")
                print("The problematic board is: ")
                print(board)
                self.restart_engine()
            except Exception as e:
                print("ok what the flip")
                print(e)
            arr.append(score)
        return np.array(arr)

    def best_moves(self, boards: List[chess.Board], color = chess.WHITE) -> List[chess.Move]:
        moves = []
        for board in boards:
            enemy_king_square = board.king(not color)
            found_king_attack = False
            if enemy_king_square:
                # if there are any ally pieces that can take king, execute one of those moves
                enemy_king_attackers = board.attackers(color, enemy_king_square)
                if enemy_king_attackers:
                    attacker_square = enemy_king_attackers.pop()
                    moves.append(chess.Move(attacker_square, enemy_king_square))
                    found_king_attack = True

            # otherwise, try to move with the stockfish chess engine
            if not found_king_attack:
                try:
                    board.turn = color
                    board.clear_stack()

                    result = self.engine.play(board, chess.engine.Limit(time=0.05))
                    moves.append(result.move)
                    continue
                except chess.engine.EngineTerminatedError:
                    print("ENGINE TERMINATED ERROR")
                    print("The board that caused this looks like:")
                    print(board)
                    #exit()
                    self.restart_engine()
                except chess.engine.EngineError:
                    print('Stockfish Engine bad state at "{}"'.format(board.fen()))
                    util.format_print_board(board)

            # if all else fails, pass
            moves.append(None)
        return moves