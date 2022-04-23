import chess
import torch

import stockfish_nn


def get_sample_tensors():
    fen = 'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R'
    board = chess.Board()
    board.set_board_fen(fen)

    batch = stockfish_nn.SparseBatch([board])
    tensors = batch.get_tensors(device=torch.device('cuda'))
    return tensors
