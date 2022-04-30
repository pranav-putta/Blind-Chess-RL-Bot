import math
import engines.base as base
from engines.base import UCBEngine, OpponentNode, PolicyEngine
from typing import List


class ExpUCB(UCBEngine):
    def __init__(self):
        super().__init__()

    def ucb(self, info_set, children: List[OpponentNode]):
        ucbs = []
        for child in children:
            ucbs.append((child.total_reward / (child.visit_count + base.eps)) + 5 * math.sqrt(
                math.log((child.availability_count / (child.visit_count + base.eps)))))
        return ucbs
