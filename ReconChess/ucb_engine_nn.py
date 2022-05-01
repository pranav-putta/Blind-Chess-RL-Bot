import math
import random

import chess

from base import UCBEngine, OpponentNode, PolicyEngine, InformationSet
from typing import List

eps = 1e-3


class PolicyNetworkUCB(UCBEngine):
    def __init__(self, policy_engine: PolicyEngine, c_puct=0.5):
        super().__init__()
        self.c_puct = c_puct
        self.policy_engine = policy_engine

    def ucb(self, info_set: InformationSet, children: List[OpponentNode]):
        N_tot = 0
        for child in children:
            N_tot += child.visit_count
        ucbs = []
        policy = self.policy_engine.generate_policy(info_set)
        board = info_set.random_sample().truth_board
        enemy_king_square = board.king(chess.BLACK)
        for i, child in enumerate(children):
            if enemy_king_square:
                if child.incoming_edge[0].to_square == enemy_king_square:
                    ucbs = [0 for i in range(len(children))]
                    ucbs[i] = 1000
                    return ucbs
            Q = child.total_reward
            P = self.policy_engine.prob_given_action(policy, child.incoming_edge[0])
            sc = (child.total_reward / (child.visit_count + eps)) + 5 * math.sqrt(
                math.log((child.availability_count / (child.visit_count + eps))))
            ucbs.append(Q + self.c_puct * P * sc + random.randrange(0, 10) ** 2)
        return ucbs
