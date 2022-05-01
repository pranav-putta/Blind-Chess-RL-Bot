import chess
from engines.eval_engine_stockfish import StockFishEvaluationEngine

from prob_board import PiecewiseGrid

import os
import random

import numpy as np
from player import Player

class LaSalleAgent(Player):
    def __init__(self):
        super().__init__()
        self.board = None
        self.color = None
        self.my_piece_captured_square = None

        self.engine = StockFishEvaluationEngine()

    def handle_game_start(self, color, board):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        :return:
        """
        self.color = color
        self.piecewisegrid = PiecewiseGrid(board)


        self.firstmove = True if color == chess.WHITE else False

    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """
        if self.firstmove:
            self.firstmove = False
            return

        """
        stockfish_vs_random = 0.9

        #num_samples = 50
        num_samples = self.piecewisegrid.num_board_states() + 1
        #print("NUM SAMPLES: " + str(num_samples))
        samples = []
        for sample in range(num_samples):
            board = self.piecewisegrid.gen_board().mirror()
            board.color = chess.BLACK
            samples.append(board)

        moves = self.engine.best_moves(samples)
        moves = [move for move in moves if not move is None]
        random_moves = []
        piece_types = []
        random_piece_types = []
        # mirrors move to opponent's side
        for i in range(len(moves)):
            move = moves[i]
            piece_types.append(samples[i].piece_at(move.from_square).piece_type)
            move.from_square = chess.square_mirror(move.from_square)
            move.to_square = chess.square_mirror(move.to_square)

            for j in range(10):
                random_moves.append(random.choice(list(samples[i].pseudo_legal_moves)))
                random_piece_types.append(samples[i].piece_at(random_moves[-1].from_square).piece_type)

            samples[i] = samples[i].mirror()

        chances = [stockfish_vs_random / len(moves) for move in moves]
        random_chances = [(1 - stockfish_vs_random) / len(random_moves) for move in random_moves]

        moves += random_moves
        chances += random_chances
        piece_types += random_piece_types
        """
        # try fitting against knight moves only
        moves = []
        knight_grid = self.piecewisegrid.piece_grids[:, :, 1] + self.piecewisegrid.piece_grids[:, :, 6]
        for file in range(8):
            for rank in range(8):
                if knight_grid[rank, file] > 0:
                    square = chess.square(file, rank)
                    moves.extend(self.knight_moves_from_square(square))

        chances = [1.0 / len(moves) for move in moves]
        piece_types = ['n'] * len(moves)

        print("OPPONENT MOVES: ")
        print(moves)
        print(self.piecewisegrid.gen_board())

        self.piecewisegrid.handle_enemy_move(list(zip(moves, piece_types, chances)), captured_piece, captured_square)

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

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        square = self.piecewisegrid.choose_sense()
        print("SENSE CHOSEN")
        print(chess.square_name(square))
        return square

    def handle_sense_result(self, sense_result):
        """
        This is a function called after your picked your 3x3 square to sense and gives you the chance to update your
        board.

        :param sense_result: A list of tuples, where each tuple contains a :class:`Square` in the sense, and if there
                             was a piece on the square, then the corresponding :class:`chess.Piece`, otherwise `None`.
        :example:
        [
            (A8, Piece(ROOK, BLACK)), (B8, Piece(KNIGHT, BLACK)), (C8, Piece(BISHOP, BLACK)),
            (A7, Piece(PAWN, BLACK)), (B7, Piece(PAWN, BLACK)), (C7, Piece(PAWN, BLACK)),
            (A6, None), (B6, None), (C8, None)
        ]
        """
        self.piecewisegrid.handle_sense_result(sense_result)

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

        # change this to depend on total uncertainty
        num_samples = self.piecewisegrid.num_board_states() + 3
        samples = []
        for sample in range(num_samples):
            board = self.piecewisegrid.gen_board()
            board.color = chess.WHITE
            samples.append(board)

        moves = self.engine.best_moves(samples)

        scores = []

        # generate new sample
        samples = []
        for sample in range(num_samples):
            samples.append(self.piecewisegrid.gen_board())

        for move in moves:
            evaluation_samples = [ board for board in samples if board.is_legal(move)]
            for board in evaluation_samples:
                board.push(move)
            score = self.engine.score_boards(samples)
            for board in evaluation_samples:
                board.pop()

            scores.append(np.sum(score))

        #print(moves)
        #print(scores)
        #print(np.argmax(scores))
        #print("THE PLAYER HAS DECIDED ON A MOVE OF: ")
        #print(moves[np.argmax(scores)])
        return moves[np.argmax(scores)]

    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        """
        This is a function called at the end of your turn/after your move was made and gives you the chance to update
        your board.

        :param requested_move: chess.Move -- the move you intended to make
        :param taken_move: chess.Move -- the move that was actually made
        :param reason: String -- description of the result from trying to make requested_move
        :param captured_piece: bool - true if you captured your opponents piece
        :param captured_square: chess.Square - position where you captured the piece
        """
        self.piecewisegrid.handle_player_move(taken_move, captured_piece)

    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        print("GAME END I THINNK")

if __name__ == "__main__":
    g = PiecewiseGrid(chess.Board())
    g.piece_grids[:, :, 4] = np.zeros((8, 8))
    g.piece_grids[3, 0, 4] = 0.0
    g.piece_grids[3, 1, 4] = 0.5
    g.piece_grids[3, 2, 4] = 0.25
    g.piece_grids[3, 3, 4] = 0.25
    g.piece_grids[4, 0, 5] = 0.25
    g.piece_grids[4, 1, 5] = 0.25
    g.piece_grids[4, 2, 5] = 0.25
    g.piece_grids[4, 3, 5] = 0.25
    print(g.num_board_states())
