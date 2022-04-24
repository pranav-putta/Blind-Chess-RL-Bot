import ReconChess.engines.base as base
import chess
import numpy as np
import torch
from collections import namedtuple
from typing import List

FeatureSet = namedtuple('FeatureSet',
                        ['num_sq', 'num_pt', 'num_planes', 'inputs', 'max_active_features', 'king_buckets'])

HalfKAv2_hm = FeatureSet(64, 11, 64 * 11, 64 * 11 * 64 / 2, 32, [-1, -1, -1, -1, 31, 30, 29, 28,
                                                                 -1, -1, -1, -1, 27, 26, 25, 24,
                                                                 -1, -1, -1, -1, 23, 22, 21, 20,
                                                                 -1, -1, -1, -1, 19, 18, 17, 16,
                                                                 -1, -1, -1, -1, 15, 14, 13, 12,
                                                                 -1, -1, -1, -1, 11, 10, 9, 8,
                                                                 -1, -1, -1, -1, 7, 6, 5, 4,
                                                                 -1, -1, -1, -1, 3, 2, 1, 0])


class SparseBatch:
    num_inputs: int
    size: int
    is_white: np.ndarray
    white: np.ndarray
    black: np.ndarray
    white_values: np.ndarray
    black_values: np.ndarray
    psqt_indices: np.ndarray
    layer_stack_indices: np.ndarray
    num_active_white_features: int
    num_active_black_features: int

    piece_types = [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING, chess.PAWN]

    def __init__(self, boards: List[chess.Board], feature_set=HalfKAv2_hm):
        self.feature_set = feature_set
        self.encode_board(boards)

    def num_pieces(self, board: chess.Board, color):
        num_pieces = 0
        for piece in self.piece_types:
            num_pieces += len(board.pieces(piece, color))
        return num_pieces

    def fill_board(self, board: chess.Board, i: int):
        self.is_white[i] = board.turn
        self.psqt_indices[i] = ((self.num_pieces(board, chess.WHITE) + self.num_pieces(board, chess.BLACK)) - 1) / 4
        self.layer_stack_indices[i] = self.psqt_indices[i]
        self.fill_features(board, i)

    def fill_features(self, board: chess.Board, i: int):
        self.num_active_black_features += self.fill_features_sparse(board, i, self.black, self.black_values,
                                                                    chess.BLACK)
        self.num_active_white_features += self.fill_features_sparse(board, i, self.white, self.white_values,
                                                                    chess.WHITE)

    def fill_features_sparse(self, board: chess.Board, i: int, features: np.ndarray, values: np.ndarray, color):
        ksq = board.king(color)
        j = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            values[i, j] = 1.0
            features[i, j] = self.feature_index(color, ksq, sq, piece)
            j += 1
        return j

    def flip_vertical(self, sq: chess.Square):
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        return chess.square(file, 7 - rank)

    def flip_horizontal(self, sq: chess.Square):
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        return chess.square(7 - file, rank)

    def orient_flip_2(self, color, sq: chess.Square, ksq: chess.Square):
        h = chess.square_file(ksq) < chess.square_file(chess.E1)
        if color == chess.BLACK:
            sq = self.flip_vertical(sq)
        if h:
            sq = self.flip_horizontal(sq)
        return sq

    def feature_index(self, color, ksq: chess.Square, sq, p: chess.Piece):
        o_ksq = self.orient_flip_2(color, ksq, ksq)
        p_idx = (p.piece_type - 1) * 2 + (p.color != color)
        if p_idx == 11:
            p_idx -= 1
        return self.orient_flip_2(color, sq, ksq) + p_idx * self.feature_set.num_sq + self.feature_set.king_buckets[
            o_ksq] * self.feature_set.num_planes

    def encode_board(self, boards: List[chess.Board]):
        self.num_inputs = self.feature_set.inputs
        self.size = len(boards)
        self.is_white = np.zeros((self.size, 1))
        self.white = np.tile(-1, (self.size, self.feature_set.max_active_features))
        self.black = np.tile(-1, (self.size, self.feature_set.max_active_features))
        self.white_values = np.tile(0.0, (self.size, self.feature_set.max_active_features))
        self.black_values = np.tile(0.0, (self.size, self.feature_set.max_active_features))
        self.psqt_indices = np.zeros(self.size)
        self.layer_stack_indices = np.zeros(self.size)

        self.num_active_white_features = 0
        self.num_active_black_features = 0

        for i, board in enumerate(boards):
            self.fill_board(board, i)

    def get_tensors(self):
        device = torch.device('cuda')
        white_values = torch.from_numpy(self.white_values).pin_memory().to(device=device, non_blocking=True).to(
            torch.float32)
        black_values = torch.from_numpy(self.black_values).pin_memory().to(device=device, non_blocking=True).to(
            torch.float32)
        white_indices = torch.from_numpy(self.white).pin_memory().to(device=device, non_blocking=True).to(torch.int32)
        black_indices = torch.from_numpy(self.black).pin_memory().to(device=device, non_blocking=True).to(torch.int32)
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).pin_memory().to(device=device,
                                                                                                          non_blocking=True).to(
            torch.float32)
        them = 1.0 - us

        psqt_indices = torch.from_numpy(
            np.ctypeslib.as_array(self.psqt_indices, shape=(self.size,))).long().pin_memory().to(device=device,
                                                                                                 non_blocking=True).to(
            torch.int64)
        layer_stack_indices = torch.from_numpy(
            np.ctypeslib.as_array(self.layer_stack_indices, shape=(self.size,))).long().pin_memory().to(device=device,
                                                                                                        non_blocking=True).to(
            torch.int64)
        return us, them, white_indices, white_values, black_indices, black_values, psqt_indices, layer_stack_indices


def evaluate(boards: List[chess.Board], model=None):
    """
    given a list of board states, generates stockfish evaluation
    :param boards:
    :param model:
    :return:
    """
    batch = SparseBatch(boards)
    tensors = batch.get_tensors()

    if model is None:
        model = torch.load('chess.pt')

    return model(*tensors)


class StockFishNNEvalEngine(base.EvaluationEngine):
    def __init__(self):
        self.model = torch.load('chess.pt')

    def evaluate_boards(self, boards: List[chess.Board]):
        batch = SparseBatch(boards)
        tensors = batch.get_tensors()
        return self.model(*tensors)
