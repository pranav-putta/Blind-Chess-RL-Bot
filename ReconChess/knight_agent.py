import copy

from player import Player
from engines.base import PolicyEngine, SenseEngine, InformationSet, SimulationEngine
from game import Game

import chess

import numpy as np
import random

class KnightAgent(Player):
    def __init__(self):
        super().__init__()

    def handle_game_start(self, color, board):
        self.sense_engine = KnightSenseEngine(color)
        self.policy_engine = KnightPolicyEngine(self.sense_engine, None, color)
        self.color = color

    def handle_opponent_move_result(self, captured_piece, captured_square):
        if captured_piece:
            file, rank = chess.square_file(captured_square), chess.square_rank(captured_square)
            if self.policy_engine.leftknightpos == (file, rank):
                self.policy_engine.leftalive = False
                if self.policy_engine.leftactive:
                    self.policy_engine.leftactive = False
            elif self.policy_engine.rightknightpos == (file, rank):
                self.policy_engine.rightalive = False
                if self.policy_engine.leftactive:
                    self.policy_engine.leftactive = True

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        return self.sense_engine.choose_sense(None)

    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        pass

    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        pass

    def handle_sense_result(self, sense_result):
        for loc in sense_result:
            if not loc[1] is None and loc[1].symbol() == 'k':
                self.sense_engine.enemykingpos = (chess.square_file(loc[0]), chess.square_rank(loc[0]))

    def choose_move(self, possible_moves, seconds_left):
        """
        Choose a move to enact from a list of possible moves.

        :param possible_moves: List(chess.Moves) -- list of acceptable moves based only on pieces
        :param seconds_left: float -- seconds left to make a move

        :return: chess.Move -- object that includes the square you're moving from to the square you're moving to
        :example: choice = chess.Move(chess.F2, chess.F4)

        :condition: If you intend to move a pawn for promotion other than Queen, please specify the promotion parameter
        :example: choice = chess.Move(chess.G7, chess.G8, promotion=chess.KNIGHT) *default is Queen
        """
        path = self.bfs_knight_to_king()
        ns = path[0].to_square
        if self.policy_engine.leftactive:
            self.policy_engine.leftknightpos = chess.square_file(ns), chess.square_rank(ns)
        else:
            self.policy_engine.rightknightpos = chess.square_file(ns), chess.square_rank(ns)
        return path[0]

    def bfs_knight_to_king(self):
        knightpos = self.policy_engine.leftknightpos if self.policy_engine.leftactive else self.policy_engine.rightknightpos
        kingpos = self.sense_engine.enemykingpos

        knightsquare = chess.square(knightpos[0], knightpos[1])
        kingsquare = chess.square(kingpos[0], kingpos[1])

        stack = [ [x] for x in self.knight_moves_from_square(knightsquare) ]
        while len(stack) > 0:
            path = stack[0]
            stack = stack[1:]

            #if len(path) >= 3:
                #temp = 12

            if path[-1].to_square == kingsquare:
                return path

            next_moves = self.knight_moves_from_square(path[-1].to_square)

            for move in next_moves:
                newpath = copy.deepcopy(path)
                newpath.append(move)
                stack.append(newpath)

    def knight_moves_from_square(self, square):
        file, rank = chess.square_file(square), chess.square_rank(square)

        knight_directions = [(2, 1), (1, 2), (-1, 2), (-2, 1), (1, -2), (2, -1), (-2, -1), (-1, -2)]

        moves = []
        for dir in knight_directions:
            upperbound = 8 if self.color == chess.WHITE else 6
            lowerbound = 2 if self.color == chess.WHITE else 0
            if file + dir[0] < 8 and file + dir[0] >= 0 and rank + dir[1] < upperbound and rank + dir[1] >= lowerbound:
                moves.append(chess.Move(square, chess.square(file + dir[0], rank + dir[1])))

        random.shuffle(moves)
        return moves

class KnightSenseEngine(SenseEngine):
    def __init__(self, color):
        self.enemykingpos = (4, 7) if color == chess.WHITE else (4, 0)

    def choose_sense(self, board: InformationSet) -> chess.Square:
        return chess.square(self.enemykingpos[0], self.enemykingpos[1])

class KnightPolicyEngine(PolicyEngine):
    def __init__(self, sense_engine: SenseEngine, eval_engine: SimulationEngine, color):
        super(KnightPolicyEngine, self).__init__(sense_engine, eval_engine)
        self.leftknightpos = (1, 0) if color == chess.WHITE else (1, 7)
        self.rightknightpos = (6, 0) if color == chess.WHITE else (6, 7)
        #self.leftactive = random.choice([True, False])
        self.leftactive = True
        self.leftalive = True
        self.rightalive = True

    def generate_policy(self, info_set: InformationSet, truth: Game) -> np.ndarray:
        pass

if __name__ == "__main__":
    print(chess.Board())
    ka = KnightAgent()
    print(ka.policy_engine.leftactive)
    print(ka.bfs_knight_to_king())