import base as base
import chess
import chess.engine
import numpy as np
from typing import List
import util
import torch
import stockfish_nn_train

EVAL_TIME_LIMIT = 0.05


class StockFishBasicEvaluationEngine(base.SimulationEngine):
    engine: chess.engine.SimpleEngine

    def __init__(self, depth=10, color=chess.WHITE):
        self.model = stockfish_nn_train.Net()
        self.model.load_state_dict(torch.load('chess.pth',map_location=torch.device('cpu')))
        self.depth = depth
        self.color = color

    def check_king_under_attack(self, board, color):
        enemy_king_square = board.king(not color)
        if enemy_king_square is not None:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = board.attackers(color, enemy_king_square)
            if enemy_king_attackers:
                return 10000 if color == self.color else -10000
            else:
                return None
        else:
            return 10000 if color == self.color else -10000

    def score_boards(self, boards: List[chess.Board]) -> np.ndarray:
        scores = []
        for board in boards:
            self_king = self.check_king_under_attack(board, self.color)
            opp_king = self.check_king_under_attack(board, not self.color)
            if self_king:
                scores.append(self_king)
                continue
            if opp_king:
                scores.append(opp_king)
                continue

            vec = stockfish_nn_train.fen_to_bit_vector(board.fen())
            scores.append(self.model(vec).detach().numpy()[0])
        return np.array(scores)

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
                except chess.engine.EngineError:
                    print('Stockfish Engine bad state at "{}"'.format(board.fen()))
                    util.format_print_board(board)

                # if all else fails, pass
                moves.append(None)
        return moves

    def close(self):
        self.engine.close()
