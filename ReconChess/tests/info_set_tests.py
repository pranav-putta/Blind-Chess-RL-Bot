import unittest
from engines import PiecewiseInformationSet, Game
import chess

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

if __name__ == '__main__':
    unittest.main()
