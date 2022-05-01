import unittest

import numpy as np

from engines import PiecewiseInformationSet, Game
import chess
from engines import move_to_feature_index, feature_output_to_move, create_policy_from_move


class InfoSetTestCases(unittest.TestCase):
    def test_knight_movement(self):
        info_set = PiecewiseInformationSet(Game())
        info_set.update_with_move((chess.Move(chess.G1, chess.H3), False))
        info_set.update_with_move((chess.Move(chess.H3, chess.G5), False))
        info_set.update_with_move((chess.Move(chess.G5, chess.H7), True))
        print(info_set.random_sample().truth_board)
        print(info_set.size())

    def test_weird(self):
        info_set = PiecewiseInformationSet(Game())
        info_set.update_with_sense([()])
        info_set.update_with_move((chess.Move(chess.F7, chess.F5), False))
        info_set.update_with_move((chess.Move(chess.B1, chess.C3), False))


class EncoderTestCases(unittest.TestCase):
    def test_a1h7_again(self):
        policy = np.zeros((8, 8, 73))
        for i in range(8):
            for j in range(8):
                for k in range(73):
                    policy[i, j, k] = 1
                    try:
                        out = feature_output_to_move(policy)
                        if out.from_square == chess.A1 and out.to_square == chess.H7:
                            print('wtf')
                            assert False
                    except:
                        pass

                    policy[i, j, k] = 0

    def test_a1h7(self):
        for sq1 in range(0, 64):
            for sq2 in range(0, 64):
                move = chess.Move(sq1, sq2)
                try:
                    encoding = create_policy_from_move(move)
                    decoding = feature_output_to_move(encoding)
                    if decoding.from_square == chess.A1 and decoding.to_square == chess.H7:
                        print('wtf')

                except:
                    pass

    def test_all(self):
        board = chess.Board()
        board.set_piece_map({})

        # try queens
        for sq in range(0, 64):
            board.set_piece_at(sq, chess.Piece(chess.QUEEN, chess.WHITE))
            for move in board.generate_legal_moves():
                encoding = create_policy_from_move(move)
                decoding = feature_output_to_move(encoding)
                assert move == decoding
            board.set_piece_map({})
        # try knights
        for sq in range(0, 64):
            board.set_piece_at(sq, chess.Piece(chess.KNIGHT, chess.WHITE))
            for move in board.generate_legal_moves():
                encoding = create_policy_from_move(move)
                decoding = feature_output_to_move(encoding)
                assert move == decoding


if __name__ == '__main__':
    unittest.main()
