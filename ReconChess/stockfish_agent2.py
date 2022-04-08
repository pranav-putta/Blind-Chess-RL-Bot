#!/usr/bin/env python3

"""
File Name:      random_agent.py
Authors:        Michael Johnson and Leng Ghuy
Date:           March 9th, 2019

Description:    Python file of a random bot
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import random
import chess
import chess.engine
import os
from player import Player

STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'


class StockfishAgent2(Player):

    def __init__(self):
        super().__init__()
        self.board = None
        self.color = None
        self.my_piece_captured_square = None

        # if STOCKFISH_ENV_VAR not in os.environ:
        #   raise KeyError(
        #      'TroutBot requires an environment variable called "{}" pointing to the Stockfish executable'.format(
        #         STOCKFISH_ENV_VAR))

        #stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        self.stockfish_path = '/usr/local/Cellar/stockfish/14.1/bin/stockfish'
        if not os.path.exists(self.stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(self.stockfish_path))

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
    
    def number_to_square(self, number):
        file = number % 8
        rank = (number - file) // 8
        return chess.square(file, rank)
    
    def square_to_number(self, square):
        return 8 * chess.square_rank(square) + chess.square_file(square)
    
    def square_parity(self, s):
        return (chess.square_file(s) + chess.square_rank(s)) % 2

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
            parity = -1
            piece_update = None
            if piece != None and piece.color != self.color:
                piece_update = piece.piece_type
                if piece.piece_type == chess.BISHOP:
                    piece_update = piece.piece_type
                    parity = (chess.square_file(square) + chess.square_rank(square)) % 2
            
            if piece_update != None:
                squares = self.board.pieces(piece.piece_type, piece.color)

                # no reason to update the board if the piece is already where we expect
                if self.board.piece_at(square) == piece:
                    continue
                    
                # if we see a piece that we didn't know was on the board, just write it in
                if len(squares) == 0:
                    self.board.set_piece_at(square, piece)

                l = list(squares)
                choice = random.choice(l)
                while choice == self.square_to_number(square) and (parity == -1 or parity != self.square_parity(self.number_to_square(choice))):
                    choice = random.choice(l)
                choice = self.number_to_square(choice)

                self.board.set_piece_at(choice, None)

                self.board.set_piece_at(square, piece)
                self.format_print_board(self.board)
            else:
                self.board.set_piece_at(square, piece)

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
        self.format_print_board(self.board)

        try:
            self.board.turn = self.color
            self.board.clear_stack()
            result = self.engine.play(self.board, chess.engine.Limit(time=0.55))
            print('Stockfish Engine lives')
            return result.move
        except chess.engine.EngineTerminatedError:
            print('Stockfish Engine died')
            #self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path, setpgrp=True)
        except chess.engine.EngineError:
            print('Stockfish Engine bad state at "{}"'.format(self.board.fen()))
            #self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path, setpgrp=True)


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

    def format_print_board(self, board):
        rows = ['8', '7', '6', '5', '4', '3', '2', '1']
        fen = board.board_fen()

        fb = "   A   B   C   D   E   F   G   H  "
        fb += rows[0]
        ind = 1
        for f in fen:
            if f == '/':
                fb += '|' + rows[ind]
                ind += 1
            elif f.isnumeric():
                for i in range(int(f)):
                    fb += '|   '
            else:
                fb += '| ' + f + ' '
        fb += '|'

        ind = 0
        for i in range(9):
            for j in range(34):
                print(fb[ind], end='')
                ind += 1
            print('\n', end='')
        print("")
