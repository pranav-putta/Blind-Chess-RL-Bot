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
from player import Player

import chess

from engines import ISMCTSPolicyEngine, PiecewiseSenseEngine, StockFishEvaluationEngine, PiecewiseInformationSet, \
    ExpUCB, NetworkPolySimEngine, PolicyNetworkUCB, StaticSenseEngine
from game import Game
import base as base
from eval_engine_network import feature_output_to_move
from util import flip_move, mirror, mirror_sense_result


class MCTSAgent(Player):

    def __init__(self, network):
        super().__init__()
        self.network = network

    def handle_game_start(self, color, board):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        """
        sense_engine = PiecewiseSenseEngine()
        sim_engine = StockFishEvaluationEngine(depth=1)
        network_policy_engine = NetworkPolySimEngine(self.network)
        ucb_engine = PolicyNetworkUCB(network_policy_engine)
        #ucb_engine = ExpUCB()
        self.engine_spec = base.EngineSpec(sense_engine, sim_engine, ucb_engine)
        self.policy_engine = ISMCTSPolicyEngine(self.engine_spec, num_iters=100)
        if color == chess.BLACK:
            board: chess.Board
            board.set_piece_at(chess.D1, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(chess.E1, chess.Piece(chess.QUEEN, chess.WHITE))
            board.set_piece_at(chess.D8, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(chess.E8, chess.Piece(chess.QUEEN, chess.BLACK))
        self.info_set = PiecewiseInformationSet(Game.new(board))
        self.color = color

    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """
        if self.color == chess.BLACK:
            captured_square = mirror(captured_square)

        # if self.firstmove:
        #    self.firstmove = False
        #   return
        """
        stockfish_vs_random = 0.9

        # num_samples = 50
        num_samples = self.info_set.size() + 1
        samples = []
        for sample in range(num_samples):
            board = self.info_set.random_sample().truth_board.mirror()
            board.color = chess.BLACK
            samples.append(board)

        moves = self.engine_spec.sim_engine.best_moves(samples)
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
            # move.from_square = chess.square(chess.square_file(move.from_square), 7 - chess.square_rank(move.from_square))
            # move.to_square = chess.square(chess.square_file(move.to_square), 7 - chess.square_rank(move.to_square))

            for j in range(10):
                random_moves.append(random.choice(list(samples[i].pseudo_legal_moves)))
                random_piece_types.append(samples[i].piece_at(random_moves[-1].from_square).piece_type)

            samples[i] = samples[i].mirror()

        chances = [stockfish_vs_random / len(moves) for move in moves]
        random_chances = [(1 - stockfish_vs_random) / len(random_moves) for move in random_moves]
        # piece_types = []
        # for board, move in zip(samples, moves):
        # piece_types.append(board.piece_at(move.from_square).piece_type)
        # piece_types = [board.piece_at(move.from_square).piece_type for board, move in zip(samples, moves)]

        # random_chances = [(1 - stockfish_vs_random) / len(random_moves) for move in random_moves]
        # random_piece_types = [board.piece_at(move.from_square).piece_type for board, move in zip(samples, random_moves)]

        moves += random_moves
        chances += random_chances
        piece_types += random_piece_types

        self.info_set.propagate_opponent_move(list(zip(moves, piece_types, chances)), captured_piece, captured_square)
        """
        self.info_set.propagate_opponent_move([], captured_piece, captured_square)
    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        sense_loc = self.engine_spec.sense_engine.choose_sense(self.info_set)
        if self.color == chess.BLACK:
            sense_loc = mirror(sense_loc)
        return sense_loc

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
        if self.color == chess.BLACK:
            sense_result = mirror_sense_result(sense_result)
        self.info_set.update_with_sense(sense_result)
        print()

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

        policy = self.policy_engine.generate_policy(self.info_set)
        move = feature_output_to_move(policy)
        if self.color == chess.WHITE:
            pass
        if self.color == chess.BLACK:
            move = flip_move(move)
        return policy, move

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
        if self.color == chess.BLACK:
            taken_move = flip_move(taken_move)
            captured_square = mirror(captured_square)
        if taken_move is not None:
            self.info_set.update_with_move((taken_move, captured_piece))
            print()

    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        self.engine_spec.sim_engine.close()
