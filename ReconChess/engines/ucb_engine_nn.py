import math

from engines.base import UCBEngine, OpponentNode, PolicyEngine, InformationSet
from typing import List


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
        for i, child in enumerate(children):
            Q = child.total_reward
            P = self.policy_engine.prob_given_action(policy, child.incoming_edge[0])
            ucbs.append(Q + self.c_puct * P * (math.sqrt(N_tot) / (1 + child.visit_count)))
        return ucbs
