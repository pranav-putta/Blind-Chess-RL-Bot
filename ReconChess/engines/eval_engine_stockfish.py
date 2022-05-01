import engines.base as base
import chess
import chess.engine
import numpy as np
from typing import List
import util

STOCKFISH_PATH = 'C:\\Users\\pputta7\\Downloads\\stockfish\\stockfish_15_x64_popcnt.exe'
EVAL_TIME_LIMIT = 0.05


class StockFishEvaluationEngine(base.SimulationEngine):
    engine: chess.engine.SimpleEngine

    def __init__(self, depth=10, color=chess.WHITE):
        self.restart_engine()
        self.depth = depth
        self.color = color

    def restart_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH, setpgrp=True)

    def check_king_under_attack(self, board, color):
        enemy_king_square = board.king(not color)
        if enemy_king_square is not None:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = board.attackers(color, enemy_king_square)
            if enemy_king_attackers:
                return 10000 if color == self.color else -10000
        else:
            return 10000 if color == self.color else -10000

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
            # for attempt in range(1, 5):
            try:
                board.clear_stack()
                score = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))[
                    'score'].relative.cp
                # score = self.engine.analyse(board, chess.engine.Limit(depth=20))['score'].relative.cp
            except AttributeError as ae:
                score = 100000
                #print("Mate found")
            except chess.engine.EngineError as e:
                print("ENGINE ERROR WHEN ATTEMPTING TO SCORE BOARD")
                print(board.fen())
                self.restart_engine()
            except Exception as e:
                print("ok what the flip")
                print(e)
            arr.append(score)
        return np.array(arr)

    def best_moves(self, boards: List[chess.Board], color=chess.WHITE) -> List[chess.Move]:
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

                    result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
                    moves.append(result.move)
                    continue
                except chess.engine.EngineTerminatedError:
                    print("ENGINE TERMINATED ERROR")
                    print("The board that caused this looks like:")
                    print(board)
                    # exit()
                    self.restart_engine()
                except chess.engine.EngineError:
                    print('Stockfish Engine bad state at "{}"'.format(board.fen()))
                    util.format_print_board(board)

                # if all else fails, pass
                moves.append(None)
        return moves

    def close(self):
        self.engine.close()
