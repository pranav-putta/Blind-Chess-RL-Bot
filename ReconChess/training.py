import chess

from engines import StockFishNNEvalEngine, RandomSenseEngine
from game import Game
from mcts_agent import MCTSAgent
from copy import copy
from engines import Net


def pit(n, newnet, oldnet):
    white_wins = 0
    for e in range(n):
        white = MCTSAgent(newnet)
        black = MCTSAgent(oldnet)
        winner, ex = play_game(white, black)
        if winner == chess.WHITE:
            white_wins += 1
    return white_wins / n


def self_play_games(n, net):
    examples = []
    for e in range(n):
        white = MCTSAgent(net)
        black = MCTSAgent(net)
        winner, ex = play_game(white, black)
        examples += ex
    return examples


def play_game(white_player: MCTSAgent, black_player: MCTSAgent):
    players = [black_player, white_player]
    examples = []

    game = Game()
    white_player.handle_game_start(chess.WHITE, chess.Board())
    black_player.handle_game_start(chess.BLACK, chess.Board())
    game.start()

    move_number = 1
    while not game.is_over():
        policy, requested_move, taken_move = play_turn(game, players[game.turn])
        if game.turn == chess.BLACK:
            print(f"\tWHITE TURN: {requested_move} [Estimated InfoSize: {white_player.info_set.size()}]")
            print(white_player.info_set.random_sample().truth_board)
            examples.append([chess.WHITE, copy(white_player.info_set.raw), policy, None])
        elif game.turn == chess.WHITE:
            print(f"\tBLACK TURN: {requested_move}")
            examples.append([chess.BLACK, copy(black_player.info_set.raw), policy, None])
        #print(game.truth_board)
        move_number += 1

    winner_color, winner_reason = game.get_winner()

    white_player.handle_game_end(winner_color, winner_reason)
    black_player.handle_game_end(winner_color, winner_reason)

    for i, example in enumerate(examples):
        reward = 1 if example[0] == winner_color else -1
        example[-1] = reward
        examples[i] = tuple(example[1:])

    return winner_color, examples


def play_turn(game, player):
    possible_moves = game.get_moves()
    possible_sense = list(chess.SQUARES)

    # notify the player of the previous opponent's move
    captured_square = game.opponent_move_result()
    player.handle_opponent_move_result(captured_square is not None, captured_square)

    # play sense action
    sense = player.choose_sense(possible_sense, possible_moves, game.get_seconds_left())
    sense_result = game.handle_sense(sense)
    player.handle_sense_result(sense_result)

    # play move action
    policy, move = player.choose_move(possible_moves, game.get_seconds_left())

    requested_move, taken_move, captured_square, reason = game.handle_move(move)
    game.took_to_long_to_move = False
    player.handle_move_result(requested_move, taken_move, reason, captured_square is not None,
                              captured_square)

    game.end_turn()
    return policy, requested_move, taken_move


def train(net, samples):
    return None


def policy_iter(load=None):
    policy_network = None

    if load is None:
        policy_network = Net()

    for iterations in range(100):
        # generate samples
        samples = self_play_games(100, policy_network)

        # run gradient descent on samples
        new_net = train(policy_network, samples)
        # play updated network against old network
        frac_win = pit(50, new_net, policy_network)
        # if updated network beats old one more than 50% of the time, update policy network
        if frac_win > 0.5:
            policy_network = new_net


policy_iter()
