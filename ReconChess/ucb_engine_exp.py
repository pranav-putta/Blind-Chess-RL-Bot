import math
import random

import base as base
from base import UCBEngine, OpponentNode, PolicyEngine
from typing import List


class ExpUCB(UCBEngine):
    def __init__(self):
        super().__init__()

    def ucb(self, info_set, children: List[OpponentNode], t):
        ucbs = []
        for child in children:
            ucbs.append((child.total_reward / (3*child.visit_count + 10)) + 15 * math.sqrt(
                math.log(1 + (t*0.1 / (child.visit_count + 1e-1)))))
            # ucbs.append((child.total_reward / (child.visit_count + base.eps)) + 5 * math.sqrt(
            # math.log((child.availability_count / (child.visit_count + base.eps)))))

        return ucbs
