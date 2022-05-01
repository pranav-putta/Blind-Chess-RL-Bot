import math
import random

import engines.base as base
from engines.base import UCBEngine, OpponentNode, PolicyEngine
from typing import List


class ExpUCB(UCBEngine):
    def __init__(self):
        super().__init__()

    def ucb(self, info_set, children: List[OpponentNode]):
        ucbs = []
        for child in children:
            try:
                ucbs.append((child.total_reward / (child.visit_count + base.eps)) + 5 * math.sqrt(
                    math.log((child.availability_count / (child.visit_count + base.eps))) + random.randrange(0, 10)))
            except:
                ucbs.append(-1)
        return ucbs
