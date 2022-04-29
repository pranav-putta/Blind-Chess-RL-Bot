import chess
import engines.base as base

import numpy as np
from typing import List, Tuple, Any, Dict, Type, Mapping
from engines.base import Player
from game import Game


class Node:
    info_set: base.InformationSet

    children: Dict
    incoming_edge: Any
    parent: Any

    visit_count: int
    total_rewards: float

    def __init__(self, info_set: base.InformationSet, incoming_edge=None, parent=None):
        self.info_set = info_set.__copy__()
        self.children = {}
        self.incoming_edge = incoming_edge
        self.parent = parent
        self.visit_count = 0
        self.total_rewards = 0

    def unexplored_children(self, game_state: Game):
        return [move for move in game_state.get_moves() if move not in self.children]

    def select_child(self, game_state: Game, engine):
        raise NotImplementedError

    def traverse(self, game_state: Game, action):
        raise NotImplementedError


class OpponentNode(Node):
    def __init__(self, info_set: base.InformationSet, incoming_edge=None, parent=None):
        super().__init__(info_set, incoming_edge, parent)

    @staticmethod
    def new(parent: Node, edge):
        move, move_result = edge
        new_node = OpponentNode(parent.info_set, move, parent)
        new_node.info_set.update_with_move(move_result)
        return new_node

    def unexplored_children(self, game_state: Game):
        pass

    def traverse(self, game_state: Game, action):
        opponent_move_result = game_state.opponent_move_result()
        edge = '_pass' if opponent_move_result is None else opponent_move_result
        if edge not in self.children:
            self.children[edge] = SelfNode.new(self, edge)
        return self.children[edge]


class SelfNode(Node):
    @staticmethod
    def new(parent: OpponentNode, incoming_edge=None):
        return SelfNode(parent.info_set, incoming_edge, parent)

    def select_child(self, game_state: Game, engine: base.SenseEngine):
        sense_location = engine.choose_sense(self.info_set)
        return sense_location

    def traverse(self, game_state: Game, sense_location: chess.Square):
        sense_result = game_state.handle_sense(sense_location)
        edge = (sense_location, tuple(sense_result))
        if edge not in self.children:
            self.children[edge] = PlayNode.new(self, sense_location, sense_result)
        return self.children[edge]


class PlayNode(Node):
    @staticmethod
    def new(parent: SelfNode, sense_location, sense_result):
        new_node = PlayNode(parent.info_set, incoming_edge=sense_location, parent=parent)
        new_node.info_set.update_with_sense(sense_result)
        return new_node

    def select_child(self, game_state: Game, engine: base.SimulationEngine):
        available_moves = game_state.get_moves()
        move = available_moves[0]

        # add some UCB calculation
        return move

    def traverse(self, game_state: Game, move):
        move_result = game_state.handle_move(move)
        edge = (move, move_result)
        if edge not in self.children:
            self.children[edge] = OpponentNode.new(self, edge)
        return self.children[edge]


class ISMCTSPolicyEngine(base.PolicyEngine):
    def __init__(self, sense_engine: base.SenseEngine,
                 eval_engine: base.SimulationEngine,
                 info_set_type: Type[base.InformationSet],
                 num_iterations: int,
                 reset_info_set_on_determinization=True):
        super().__init__(sense_engine, eval_engine)
        initial_shape = (100, 64 * (7 * 8 + 8 + 9))
        self.info_set_type = info_set_type
        self.Q = np.zeros(initial_shape)
        self.N = np.zeros(initial_shape)
        self.P = np.zeros(initial_shape)
        self.visited: List[Node] = []
        self.trees = []
        self.num_iters = num_iterations
        self.reset_info_set_on_determinization = reset_info_set_on_determinization

    def generate_policy(self, player_info_set: base.InformationSet, other_info_set: base.InformationSet) -> np.ndarray:
        # TODO: fix monte carlo call
        return self.mcts(player_info_set, other_info_set)

    def ucb(self, move):
        return 0

    def mcts(self, p1_info_set: base.InformationSet, p2_info_set: base.InformationSet):
        player = 0
        self.trees = [SelfNode(p1_info_set), OpponentNode(p2_info_set)]

        for i in range(self.num_iters):
            # perform initial determinization
            trees = self.trees.copy()
            determinization = p1_info_set.random_sample()
            trees, game_state = self.select(trees, player, determinization)
            if not game_state.is_over():
                self.expand(trees, player, game_state)
            rewards = self.simulate(trees)
            self.backprop(trees, rewards)

        policy = sorted(self.trees[player].children, key=lambda child: child.total_rewards)
        return map(lambda node: node.incoming_edge, policy)

    def backprop(self, trees: List[Node], rewards):
        for player in range(len(trees)):
            trees[player].visit_count += 1
            trees[player].total_rewards += rewards[player]
            trees[player] = trees[player].parent

    def simulate(self, game_state):
        reward = self.eval_engine.score_boards([game_state])
        return [reward, -reward]

    def expand(self, trees: List[Node], player, game_state: Game):
        if game_state.is_over():
            return trees, game_state

        sense_location = trees[player].select_child(game_state, self.sense_engine)
        trees[player] = trees[player].traverse(game_state, sense_location)

        move_action = trees[player].select_child(game_state, self.eval_engine)

        for p in [player, not player]:
            trees[p] = trees[p].traverse(game_state, move_action)

        return trees

    def select(self, trees: List[Node], player, game_state: Game):
        if game_state.is_over():
            return trees, game_state
        for tree in trees:
            if len(tree.unexplored_children(game_state)) > 0:
                return trees, game_state

        # sense action
        sense_action = trees[player].select_child(game_state, self.sense_engine)
        trees[player] = trees[player].traverse(game_state, sense_action)

        # move action
        move_action = trees[player].select_child(game_state, self.eval_engine)

        # propagate player trees to the next state
        for p in [player, not player]:
            trees[p] = trees[p].traverse(game_state, move_action)
        return self.select(trees, not player, game_state)
