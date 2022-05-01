import chess
import torch.utils.data
from torch import optim

from engines import StockFishNNEvalEngine, RandomSenseEngine
from game import Game
from mcts_agent import MCTSAgent
from copy import copy
from engines import Net
import pickle
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import numpy as np

MAX_MOVES = 100


def pit(n, newnet, oldnet):
    white_wins = 0
    for e in range(n):
        white = MCTSAgent(newnet)
        black = MCTSAgent(oldnet)
        winner, ex = play_game(white, black)
        if winner == chess.WHITE:
            white_wins += 1
    return white_wins / n


def self_play_games(n, net, name=''):
    examples = []
    for e in range(n):
        print(f"GAME {e}")
        white = MCTSAgent(net)
        black = MCTSAgent(net)
        winner, ex = play_game(white, black)
        examples += ex
    pickle.dump(examples, open(name, 'wb'))
    return examples


def play_game(white_player: MCTSAgent, black_player: MCTSAgent):
    players = [black_player, white_player]
    examples = []

    game = Game()
    white_player.handle_game_start(chess.WHITE, chess.Board())
    black_player.handle_game_start(chess.BLACK, chess.Board())
    game.start()

    move_number = 1
    turn = chess.WHITE
    while not game.is_over() and move_number < MAX_MOVES:
        game.turn = turn
        print('WHITE TURN' if game.turn == chess.WHITE else 'BLACK TURN')
        policy, requested_move, taken_move = play_turn(game, players[game.turn])
        if game.turn == chess.BLACK:
            print(white_player.info_set.random_sample().truth_board)
            examples.append([chess.WHITE, copy(white_player.info_set.raw), policy, None])
        elif game.turn == chess.WHITE:
            print(black_player.info_set.random_sample().truth_board)
            examples.append([chess.BLACK, copy(black_player.info_set.raw), policy, None])
        print("TRUE BOARD")
        print(game.truth_board)
        move_number += 1
        turn = not turn

    if game.is_over():
        winner_color, winner_reason = game.get_winner()
    else:
        winner_color, winner_reason = None, None

    white_player.handle_game_end(winner_color, winner_reason)
    black_player.handle_game_end(winner_color, winner_reason)

    for i, example in enumerate(examples):
        if winner_color is None:
            reward = 0
        else:
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
    print(f"Sense = {chess.square_name(sense)}")
    sense_result = game.handle_sense(sense)
    player.handle_sense_result(sense_result)

    # play move action
    policy, move = player.choose_move(possible_moves, game.get_seconds_left())

    try:
        requested_move, taken_move, captured_square, reason = game.handle_move(move)
        print(f"Move: {requested_move}, Taken_Move: {taken_move}, Captured_Square: {captured_square}, Color: {game.turn}")

    except Exception as e:
        print(e)
        requested_move, taken_move, captured_square, reason = move, None, None, None

    game.took_to_long_to_move = False

    player.handle_move_result(requested_move, taken_move, reason, captured_square is not None,
                              captured_square)
    game.end_turn()
    return policy, requested_move, taken_move


class ChessDataset(Dataset):
    def __init__(self, examples):
        info_set, policy, value = zip(*examples)
        self.inputs = info_set
        self.policy = list(policy)
        self.values = list(value)

    def __getitem__(self, item):
        return self.inputs[item], (self.policy[item], self.values[item])

    def __len__(self):
        return len(self.inputs)


def loss_func(outputs, policy, z):
    v_theta, p_theta = outputs
    policy = policy.flatten().reshape(10, -1)
    val= (v_theta - z) ** 2 - torch.sum(torch.log(p_theta) * policy, dim=1)
    return val


def train(net, samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset = ChessDataset(samples)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
    criterion = loss_func
    optimizer = optim.AdamW(net.parameters())

    net = net.to(device)
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, (policies, values) = data
            inputs = inputs.to(device)
            policies = policies.to(device)
            values = values.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, policies, values)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                # denominator for loss should represent the number of positions evaluated
                # independent of the batch size
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (2000 * len(policies))))
                running_loss = 0.0

    print('Finished Training')

    PATH = 'chess.pth'
    torch.save(net.state_dict(), PATH)

    print('Evaluating model')


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


net = Net()


def self_play(n):
    self_play_games(3, net, name=f'examples/example{n}.pkl')


#samples = pickle.load(open('example1.pkl', 'rb'))
#train(net, samples)


if __name__ == '__main__':
    pool = mp.Pool(48)
    pool.map(self_play, range(48))