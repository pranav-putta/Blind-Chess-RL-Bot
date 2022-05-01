#!/usr/bin/env python3

"""
File Name:      naive_stockfish_agent.py
Authors:        Pranav Putta and Erik Scarlatescu
Date:           March 9th, 2019

Description:    Python file of naive stockfish bot
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import random
import chess
import chess.engine
import os
import util
from player import Player

STOCKFISH_PATH = 'C:\\Users\\escarlatescu3\\OneDrive - Georgia Institute of Technology\\Desktop\\stockfish_15_x64_avx2.exe'

class StockfishAgent2(Player):

    def __init__(self):
        super().__init__()
        self.board = None
        self.color = None
        self.my_piece_captured_square = None

        self.stockfish_path = STOCKFISH_PATH
        if not os.path.exists(self.stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(self.stockfish_path))
        self.engine = None
        self.restart_engine()

    def restart_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path, setpgrp=True)

    def handle_game_start(self, color, board: chess.Board):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        """
        self.board = board
        self.color = color

    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """
        self.my_piece_captured_square = captured_square
        if captured_piece:
            self.board.remove_piece_at(captured_square)

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        if self.my_piece_captured_square:
            return self.my_piece_captured_square

        # if we might capture a piece when we move, sense where the capture will occur
        future_move = self.choose_move(possible_moves, seconds_left)
        if future_move is not None and self.board.piece_at(future_move.to_square) is not None:
            return future_move.to_square

        # otherwise, just randomly choose a sense action, but don't sense on a square where our pieces are located
        for square, piece in self.board.piece_map().items():
            if piece.color == self.color:
                possible_sense.remove(square)
        return random.choice(possible_sense)

    def naive_replace_piece(self, square, piece):
        """
        Naively removes a piece from the board belief state without breaking the stockfish engine.
        Stockfish only suggests moves if the board is in a valid state. If for example, we have 2 kings or 9 pawns,
        stockfish engine will crash.

        Naively, if we sense the board and observe pieces missing where we thought they would be, just remove them.
        Unless that piece is the king. Stockfish crashes when the kings aren't present. So, relocate the king to the nearest location.
        :param square: square to remove
        :param piece: piece we're removing
        :return:
        """
        current_piece = self.board.piece_at(square)
        if current_piece is not None and current_piece.color != self.color and current_piece == chess.KING:
            print("relocating king")
            self.relocate_king(square)
        else:
            self.board.set_piece_at(square, piece)

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
        for square, piece in sense_result:
            if piece is None or piece.color == self.color:
                self.naive_replace_piece(square, piece)
                continue
            # no reason to update the board if the piece is already where we expect
            if self.board.piece_at(square) == piece:
                continue

            # if we see a piece that we didn't know was on the board, randomly remove extra piece, and set the
            # observed piece
            squares = list(self.board.pieces(piece.piece_type, piece.color))
            if len(squares) > 0:
                remove_square = random.choice(squares)
                self.naive_replace_piece(remove_square, None)
            self.naive_replace_piece(square, piece)

    def relocate_king(self, square):
        if square is None:
            return
        centerx, centery = chess.square_file(square), chess.square_rank(square)
        for radius in range(1, 9):
            for x in range(centerx - radius, centerx + radius + 1):
                if self.board.piece_at(chess.square(x, centery - radius)) is None:
                    self.set_piece(x, centery - radius, chess.KING)
                    return
                if self.board.piece_at(chess.square(x, centery + radius)) is None:
                    self.set_piece(x, centery + radius, chess.KING)
                    return

            for y in range(centery - radius + 1, centery + radius - 1):
                if self.board.piece_at(centerx - radius, y) is None:
                    self.set_piece(centerx - radius, y, chess.KING)
                    return
                if self.board.piece_at(centerx + radius, y) is None:
                    self.set_piece(centerx + radius, y, chess.KING)
                    return

    def set_piece(self, x, y, piece):
        if 0 <= x < 8 and 0 <= y < 8:
            self.board.set_piece_at(chess.square(x, y), piece)

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
        enemy_king_square = self.board.king(not self.color)
        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                return chess.Move(attacker_square, enemy_king_square)

        # otherwise, try to move with the stockfish chess engine

        try:
            self.board.turn = self.color
            self.board.clear_stack()

            if len(self.board.pieces(chess.KING, not self.color)) == 0:
                # this was an L, tried to attack the king and it wasn't there! Relocate it to a new location.
                print('relocating the king')
                self.relocate_king(enemy_king_square)
            result = self.engine.play(self.board, chess.engine.Limit(time=0.5))
            print('Stockfish Engine lives')
            util.format_print_board(self.board, force_verbose=False)
            return result.move
        except chess.engine.EngineTerminatedError:
            print('Stockfish Engine died')
            util.format_print_board(self.board, force_verbose=True)
            exit()
            self.restart_engine()
        except chess.engine.EngineError:
            print('Stockfish Engine bad state at "{}"'.format(self.board.fen()))
            util.format_print_board(self.board)

        # if all else fails, pass
        return None

    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        """
        This is a function called at the end of your turn/after your move was made and gives you the chance to update
        your board.

        :param requested_move: chess.Move -- the move you intended to make
        :param taken_move: chess.Move -- the move that was actually made
        :param reason: String -- description of the result from trying to make requested_move
        :param captured_piece: bool -- true if you captured your opponents piece
        :param captured_square: chess.Square -- position where you captured the piece
        """
        if taken_move is not None:
            self.board.push(taken_move)

    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        try:
            # if the engine is already terminated then this call will throw an exception
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass
