from torch import nn
import torch
import torch.nn.functional as F
from engines.base import SimulationEngine, PolicyEngine, InformationSet
from typing import List
import chess
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32 * 64, 832)
        self.fc2 = nn.Linear(832, 416)
        self.fc3 = nn.Linear(416, 208)
        self.fc4 = nn.Linear(208, 104)
        self.fc_out_score = nn.Linear(104, 1)
        self.fc_out_policy = nn.Linear(104, 4672)

    def forward(self, x):
        batch = x.shape[0]
        x = torch.tensor(x.flatten().reshape(batch, -1), dtype=torch.float)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        score = self.fc_out_score(x)
        policy = self.fc_out_policy(x)
        return score, policy



def move_to_feature_index(move: chess.Move):
    fr, to = move.from_square, move.to_square
    i, j = chess.square_rank(fr), chess.square_file(fr)
    dx, dy = chess.square_file(to) - j, chess.square_rank(to) - i

    direction = 0
    dist = 0
    if dx == 0 or dy == 0:  # solid direction
        direction += 4
        if abs(dy) > 0:  # N/S or E/W
            direction += 2
        if dx > 0 or dy > 0:  # N or E
            direction += 1
        dist = max(abs(dx), abs(dy))
    elif abs(dx) == abs(dy):  # diagonal direction
        if dy > 0:  # NE/NW or SE/SW
            direction += 2
        if dx > 0:
            direction += 1
        dist = abs(dx)
    else:  # knight
        if abs(dx) > abs(dy):
            direction += 4
        if dx > 0:
            direction += 2
        if dy > 0:
            direction += 1
        dist = 8
    k = (dist - 1) * 8 + direction
    return i, j, k


knight_enc = [(-1, -2), (-1, 2), (1, -2), (1, 2), (-2, -1), (-2, 1), (2, -1), (2, 1)]
direction_enc = [(-1, -1), (1, -1), (-1, 1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]


def feature_output_to_move(policy: np.ndarray):
    if policy.sum() == 0:
        return None
    from_rank, from_file, opt = np.unravel_index(np.argmax(policy), policy.shape)

    if opt > 63:
        raise NotImplementedError
    elif opt // 8 == 7:  # knight
        to_file, to_rank = knight_enc[opt % 8]
        to_rank += from_rank
        to_file += from_file
    else:  # regular move
        distance = 1 + (opt // 8)
        direction = direction_enc[opt % 8]
        df, dr = direction[0] * distance, direction[1] * distance
        to_rank, to_file = from_rank + dr, from_file + df
    return chess.Move(chess.square(from_file, from_rank), chess.square(to_file, to_rank))


def create_policy_from_move(move: chess.Move):
    policy = np.zeros((8, 8, 73))
    policy[move_to_feature_index(move)] = 1
    return policy


class NetworkPolySimEngine(SimulationEngine, PolicyEngine):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def score_boards(self, boards: List[chess.Board]):
        return 0

    def generate_policy(self, self_info_set: InformationSet):
        score, policy = self.net(self_info_set.raw[None,:])
        return policy.reshape(8, 8, 73)

    def prob_given_action(self, policy, move):
        return policy[move_to_feature_index(move)].item()

    def generate_policies(self, self_info_sets: List[InformationSet]) -> np.ndarray:
        input = []
        for info_set in self_info_sets:
            input.append(info_set.raw)
        input = np.array(input)
        score, policy = self.net(input)
        return policy.reshape(8, 8, 73)
