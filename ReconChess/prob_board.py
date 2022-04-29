import chess
import numpy as np

import random

# to convolve uncertainties
from scipy import signal

order = [4, 20, 3, 19, 7, 23, 0, 16, 6, 22, 1, 17, 5, 21, 2, 18, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30,
         15, 31]

import copy

class PiecewiseGrid:
    def __copy__(self):
        newgrid = PiecewiseGrid()
        newgrid.piece_grids = np.copy(self.piece_grids)
        newgrid.piece_types = copy.deepcopy(self.piece_types)
        newgrid.own_pieces = copy.deepcopy(self.own_pieces)
        newgrid.captured = copy.deepcopy(self.captured)
        newgrid.promoted = copy.deepcopy(self.promoted)

    def __init__(self, board: chess.Board):
        self.piece_grids = np.zeros((8, 8, 32))
        self.piece_types = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'] + ['p'] * 8
        self.piece_types = self.piece_types + [x.upper() for x in self.piece_types]

        self.own_pieces = [False] * 16 + [True] * 16
        self.captured = [False] * 32
        self.promoted = [False] * 32

        for i in range(32):
            piece = chess.Piece.from_symbol(self.piece_types[i])
            if piece != None:
                pieces = list(board.pieces(piece.piece_type, piece.color))
                file = chess.square_file(pieces[0])
                rank = chess.square_rank(pieces[0])
                self.piece_grids[rank, file, i] = 1.0
                board.set_piece_at(pieces[0], None)
            else:
                self.captured[i] = True

        # generates standard board programmatically
        """
        for p in range(2):
            for y in range(2):
                for x in range(8):
                    self.piece_grids[p * y + (1 - p) * (7 - y), x, 16 * p + 8 * y + x] = 1.0
        """

        # rook
        self.piece_grids[:, :, 0] = np.zeros((8, 8))
        self.piece_grids[4, 2, 0] = 0.2
        self.piece_grids[5, 5, 0] = 0.1
        self.piece_grids[5, 6, 0] = 0.1
        self.piece_grids[4, 5, 0] = 0.1
        self.piece_grids[4, 7, 0] = 0.1
        self.piece_grids[3, 5, 0] = 0.1
        self.piece_grids[3, 6, 0] = 0.1
        self.piece_grids[2, 6, 0] = 0.2

        # knight
        self.piece_grids[:, :, 1] = np.zeros((8, 8))
        self.piece_grids[2, 4, 1] = 1.0
        self.piece_grids[3, 3, 1] = 0.3
        self.piece_grids[3, 4, 1] = 0.1
        self.piece_grids[2, 4, 1] = 0.2
        self.piece_grids[2, 5, 1] = 0.1
        self.piece_grids[2, 6, 1] = 0.3

        # pawn
        self.piece_grids[:, :, 11] = np.zeros((8, 8))
        self.piece_grids[4, 3, 11] = 1.0

    # possible moves represents a probability distribution. it is stored as a list of tuples of the form (move, piece_type, chance)
    # move chances are stored as numpy array
    def update_prob_board_from_moves(self, possible_moves):
        for move, piece_type, chance in possible_moves:
            # todo: add possibility of pawn switching columns
            # self.piece_types = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'] + ['p'] * 8

            move_indices = []
            if piece_type == 'r':
                move_indices.append(0)
                move_indices.append(7)
            elif piece_type == 'n':
                move_indices.append(1)
                move_indices.append(6)
            elif piece_type == 'b':
                move_indices.append(2)
                move_indices.append(5)
            elif piece_type == 'q':
                move_indices.append(3)
            elif piece_type == 'k':
                move_indices.append(4)
            else:  # pawn
                file = chess.square_file(move.from_square)
                move_indices.append(8 + file)

            for move_index in move_indices:
                from_file = chess.square_file(move.from_square)
                from_rank = chess.square_rank(move.from_square)

                to_file = chess.square_file(move.to_square)
                to_rank = chess.square_rank(move.to_square)

                prob = self.piece_grids[from_rank, from_file, move_index] * chance
                self.piece_grids[from_rank, from_file, move_index] -= prob
                self.piece_grids[to_rank, to_file, move_index] += prob

    # possible moves represents a probability distribution. it is stored as a list of tuples of the form (move, piece_type, chance)
    def handle_enemy_move(self, possible_moves, captured_piece: bool, captured_square: chess.Square):
        self.piece_grids_temp = self.piece_grids.copy()
        self.enemy_moves = possible_moves
        if captured_piece:
            file = chess.square_file(captured_square)
            rank = chess.square_rank(captured_square)

            piece_chances = np.array([0.0] * 32)
            board = self.gen_certain_board()

            for i in range(16):
                piece_grid = self.piece_grids[:, :, i]
                if self.captured[i]:
                    continue

                if self.piece_types[i] == 'r' or self.piece_types[i] == 'q':
                    x = file
                    while x < 8 and board.piece_at(chess.square(x, rank)) == None:
                        piece_chances[i] += piece_grid[rank, x]
                        x += 1
                    x = file
                    while x >= 0 and board.piece_at(chess.square(x, rank)) == None:
                        piece_chances[i] += piece_grid[rank, x]
                        x -= 1

                    y = rank
                    while y < 8 and board.piece_at(chess.square(file, y)) == None:
                        piece_chances[i] += piece_grid[y, file]
                        y += 1

                    y = rank
                    while y >= 0 and board.piece_at(chess.square(file, y)) == None:
                        piece_chances[i] += piece_grid[y, file]
                        y -= 1

                if self.piece_types[i] == 'b' or self.piece_types[i] == 'q':
                    i = 0
                    while file + i < 8 and rank + i < 8 and board.piece_at(chess.square(file + i, rank + i)) == None:
                        piece_chances[i] += piece_grid[rank + i, file + i]
                        i += 1
                    i = 0
                    while file - i >= 0 and rank + i < 8 and board.piece_at(chess.square(file - i, rank + i)) == None:
                        piece_chances[i] += piece_grid[rank + i, file - i]
                        i += 1
                    i = 0
                    while file - i >= 0 and rank - i >= 0 and board.piece_at(chess.square(file - i, rank - i)) == None:
                        piece_chances[i] += piece_grid[rank - i, file - i]
                        i += 1
                    i = 0
                    while file + i < 8 and rank - i >= 0 and board.piece_at(chess.square(file + i, rank - i)) == None:
                        piece_chances[i] += piece_grid[rank - i, file + i]
                        i += 1

                if self.piece_types[i] == 'n':
                    if file + 2 < 8 and rank + 1 < 8:
                        piece_chances[i] += piece_grid[rank + 1, file + 2]
                    if file + 1 < 8 and rank + 2 < 8:
                        piece_chances[i] += piece_grid[rank + 2, file + 1]

                    if file - 1 >= 0 and rank + 2 < 8:
                        piece_chances[i] += piece_grid[rank + 2, file - 1]
                    if file - 2 >= 0 and rank + 1 < 8:
                        piece_chances[i] += piece_grid[rank + 1, file - 2]

                    if file + 1 < 8 and rank - 2 >= 0:
                        piece_chances[i] += piece_grid[rank - 2, file + 1]
                    if file + 2 < 8 and rank - 1 >= 0:
                        piece_chances[i] += piece_grid[rank - 1, file + 2]

                    if file - 2 >= 0 and rank - 1 >= 0:
                        piece_chances[i] += piece_grid[rank - 1, file - 2]
                    if file - 1 >= 0 and rank - 2 >= 0:
                        piece_chances[i] += piece_grid[rank - 2, file - 1]

                if self.piece_types[i] == 'p':
                    piece_chances[i] += piece_grid[rank - 1, file - 1]
                    piece_chances[i] += piece_grid[rank - 1, file + 1]

            piece_chances /= np.sum(piece_chances)
            piece_chances.reshape(32, 1)

            self.piece_grids = self.piece_grids * (1 - piece_chances)

            piece_mask = np.ones((8, 8))
            piece_mask[file, rank] = 0
            piece_mask = np.repeat(piece_mask[:, :, np.newaxis], 32, axis=2)

            self.piece_grids = self.piece_grids * piece_mask

            for i in range(32):
                self.piece_grids[rank, file, i] = piece_chances[i]
        else:
            self.update_prob_board_from_moves(self.enemy_moves)

    def choose_sense(self):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        KING_ATTACK = 0.25
        PIECE_PIN = 0.15

        self.uncertainty = 0.5 - np.abs(0.5 - self.piece_grids)
        self.uncertainty = np.zeros((8, 8, 32))

        board = self.gen_certain_board()

        # find location of your own king
        rank, file = np.unravel_index(np.argmax(self.piece_grids[:, :, 4]), (8, 8))

        # add uncertainty from knight attacks
        knights = [self.piece_types[i] == 'n' and not self.captured[i] for i in range(32)]
        knight_directions = [(2, 1), (1, 2), (-1, 2), (-2, 1), (1, -2), (2, -1), (-2, -1), (-1, -2)]
        for dir in knight_directions:
            x = file + dir[0]
            y = rank + dir[1]
            if x < 8 and y < 8 and x >= 0 and y >= 0 and board.piece_at(chess.square(x, y) == None):
                self.uncertainty[x, y, knights] += KING_ATTACK

        # add uncertainty from sliding attacks
        straight_attackers = [(self.piece_types[i] == 'q' or self.piece_types[i] == 'r') and not self.captured[i] for i in range(32)]
        diagonal_attackers = [(self.piece_types[i] == 'q' or self.piece_types[i] == 'b') and not self.captured[i] for i in range(32)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for i, dir in enumerate(directions):
            x = file + dir[0]
            y = rank + dir[1]
            num_hits = 0

            while x < 8 and x >= 0 and y < 8 and y >= 0 and num_hits < 2:
                piece = board.piece_at(chess.square(x, y))
                if piece != None:
                    num_hits += 1 if piece.color == chess.WHITE else 2
                else:
                    self.uncertainty[y, x, straight_attackers if i < 4 else diagonal_attackers] += KING_ATTACK if num_hits == 0 else PIECE_PIN
                x += dir[0]
                y += dir[1]

        #return self.uncertainty
        self.uncertainty = np.max(self.uncertainty, axis=2)

        # finds which 3x3 squares have the highest uncertainties
        self.uncertainties = signal.convolve2d(self.uncertainty, np.ones((3, 3)), mode="same")

        rank, file = np.unravel_index(np.argmax(self.uncertainties), (8, 8))
        return chess.square(file, rank)

        # return self.uncertainties[1:7, 1:7]

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

        if len(self.enemy_moves) > 0:
            probs = self.piece_grids_temp.sum(axis=2)
            maxes = self.piece_grids_temp.max(axis=2)

            # prune enemy moves that are no longer possible
            # might not be the most efficient algorithm for this, but it works for now
            for loc in sense_result:
                file = chess.square_file(loc[0])
                rank = chess.square_rank(loc[0])

                # if there was no piece in the square previously and there is one there now, we know that piece moved
                if probs[rank, file] < 0.001 and loc[1] != None:
                    piece_type = loc[1].symbol()
                    self.enemy_moves = [x for x in self.enemy_moves if x[1] == piece_type]

                # if we know where a piece is for certain, we can make some inferences
                if maxes[rank, file] > 0.99:
                    piece_index = np.argmax(self.piece_grids[rank, file, :])
                    piece_type = self.piece_types[piece_index]

                    if loc[1] == None:  # if it's no longer there, it must have moved
                        self.enemy_moves = [x for x in self.enemy_moves if x[1] == piece_type]
                    elif loc[1].symbol() == piece_type:  # if it's still there, it can't have moved
                        self.enemy_moves = [x for x in self.enemy_moves if x[1] != piece_type]

            self.piece_grids = self.piece_grids_temp

            # renormalize move probabilities
            moves, piece_types, chances = zip(*self.enemy_moves)
            chances = np.array(chances)
            chances /= np.sum(chances)
            self.enemy_moves = zip(moves, piece_types, chances.tolist())

            if len(self.enemy_moves) == 0:
                print("WE SHOULD NOT BE HERE")

        # make a second update to probability grid with more informed enemy moves
        self.update_prob_board_from_moves(self.enemy_moves)

        # update piece grid based on sense results
        handled = [False] * 32
        for loc in sense_result:
            file = chess.square_file(loc[0])
            rank = chess.square_rank(loc[0])

            self.piece_grids[rank, file, :] *= 0

            # if there is a piece, find which one it is and update probabilities accordingly
            if loc[1] != None:
                """
                piece_index = []
                for i,x in enumerate(self.piece_types):
                    if x == loc[1].symbol() and not handled[i]:
                        piece_index.append(i)
                piece_index = piece_index[0]
                """

                piece_index = [i for i, x in enumerate(self.piece_types) if x == loc[1].symbol() and not handled[i]][0]
                self.piece_grids[:, :, piece_index] = np.zeros((8, 8))
                self.piece_grids[rank, file, piece_index] = 1
                handled[piece_index] = True

            # todo: pawns need more logic to handle duplicates
        divider = self.piece_grids.sum((0, 1)).reshape(1, 1, 32)
        divider[divider < 0.001] = 1
        print(self.piece_types[12])
        print(self.piece_grids.sum((0, 1)).reshape(1, 1, 32))
        self.piece_grids /= divider

    def gen_certain_board(self):
        board = chess.Board()
        board.set_piece_map({})
        for i in range(32):
            piece_grid = self.piece_grids[:, :, i]
            if np.max(piece_grid) > 0.99:
                location = np.argmax(piece_grid)
                board.set_piece_at(location, chess.Piece.from_symbol(self.piece_types[i]))

        return board

    def gen_board(self):
        spaces = []
        board = chess.Board()
        board.set_piece_map({})

        for i in order:
            # transposing from numpy format to board format begins here
            piece_grid = np.copy(self.piece_grids[:, :, i]).flatten()

            if np.sum(piece_grid) < 0.001:
                continue

            piece_grid[spaces] = 0
            piece_grid /= np.sum(piece_grid)

            num = np.random.choice(64, 1, p=piece_grid)[0]
            # and ends here when flattened index is reinterpreted as board index
            board.set_piece_at(num, chess.Piece.from_symbol(self.piece_types[i]))

            if i == 0 or i == 1 or i == 11:
                print(self.piece_types[i])
                print(self.piece_grids[:, :, i])
                print("SUM: " + str(np.sum(self.piece_grids[:, :, i])))

            spaces.append(num)

        return board


# This class hasn't been fleshed out yet. Its purpose will be to have the PiecewiseGrid interface with the reconchess game
def PiecewisePlayer(Player):
    def __init__(self):
        pass

    def handle_game_start(self, color, board):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        :return:
        """
        # TODO: implement this method
        pass

    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """
        pass

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        # TODO: update this method
        return random.choice(possible_sense)

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
        # TODO: implement this method
        # Hint: until this method is implemented, any senses you make will be lost.
        pass

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
        # TODO: update this method
        choice = random.choice(possible_moves)
        return choice

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
        # TODO: implement this method
        pass

    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        # TODO: implement this method
        pass


b = chess.Board()
# print(b)
# b.set_piece_map({})
# print(b)

g = PiecewiseGrid(b)
#print(g.gen_board())
#print(g.gen_certain_board())
g.gen_board()
g.handle_enemy_move([], True, chess.square(6, 4))
print("HANDLING MOVE")
b = g.gen_board()
print(b)
print("ABOUT TO CHOOSE SENSE")
square = g.choose_sense()

file = chess.square_file(square)
rank = chess.square_rank(square)
print("SENSE SQUARE: " + str(file) + ", " + str(rank))

sense_result = []
for x in range(file - 1, file + 2):
    for y in range(rank - 1, rank + 2):
        sense_result.append((chess.square(x, y), b.piece_at(chess.square(x, y))))

g.handle_sense_result(sense_result)

b2 = g.gen_board()
print(g.piece_grids[:, :, 0])
print(g.piece_grids[:, :, 1])

# construct sense
