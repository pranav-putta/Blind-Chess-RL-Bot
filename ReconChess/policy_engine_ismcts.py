import random

import chess
import numpy as np

import base as base

from collections import namedtuple
from typing import List, Tuple, Any, Dict, Type, Mapping
from game import Game
import math
# from tqdm import tqdm
from eval_engine_network import move_to_feature_index, feature_output_to_move
from info_set_piecewise import PiecewiseInformationSet

eps = 1e-10


class ISMCTSPolicyEngine(base.PolicyEngine):
    def __init__(self, player_engine_spec: base.EngineSpec, opponent_engine_spec=None, white=True, num_iters=10):
        super().__init__()
        if opponent_engine_spec is None:
            self.engine_specs = [player_engine_spec, player_engine_spec]
        else:
            self.engine_specs = [player_engine_spec, opponent_engine_spec]
        self.trees = []
        self.white = white
        self.num_iters = num_iters

    def sense_engine(self, player):
        return self.engine_specs[player].sense_engine

    def sim_engine(self, player):
        return self.engine_specs[player].sim_engine

    def ucb_engine(self, player):
        return self.engine_specs[player].ucb_engine

    def new_game(self, board):
        g = Game()
        g.truth_board = board.copy()
        g.white_board = board.copy()
        g.black_board = board.copy()
        return g

    def score_board(self, board, num_iters):
        tmp = self.num_iters
        self.num_iters = num_iters
        player_info_set = PiecewiseInformationSet(self.new_game(board))
        other_info_set = player_info_set.__copy__()
        self.mcts(player_info_set, other_info_set)
        self.num_iters = tmp
        return self.trees[0].total_reward

    def generate_policy(self, player_info_set: base.InformationSet):
        player_info_set.propagate_opponent_move([], None, False)

        other_info_set = player_info_set.random_sample()
        other_info_set = PiecewiseInformationSet(other_info_set)

        self.mcts(player_info_set.__copy__(), other_info_set)
        policy = np.ones((8, 8, 73))
        policy *= float('-inf')

        if len(self.trees[0].children) == 0:
            # mf reached a terminal game state so don't make any moves
            return policy
        best_sense = max(self.trees[0].children.values(), key=lambda child: child.total_reward)
        moves = sorted(best_sense.children.values(), key=lambda child: child.total_reward, reverse=True)

        for node in moves:
            node: base.OpponentNode
            if node.incoming_edge[1] is not None:
                policy[move_to_feature_index(node.incoming_edge[1])] = node.total_reward
            else:
                print(node.incoming_edge)

        return policy

    def get_turn(self, player):
        if player == 0:
            return self.white
        else:
            return not self.white

    def mcts(self, p1_info_set: base.InformationSet, p2_info_set: base.InformationSet):
        player = 0
        self.trees = [base.SelfNode(p1_info_set, chess.WHITE, incoming_edge=None, parent=None),
                      base.OpponentNode(p2_info_set, chess.BLACK, incoming_edge=None, parent=None)]

        for i in range(self.num_iters):
            # perform initial determinization
            trees = self.trees.copy()
            determinization = p1_info_set.random_sample()
            trees, game_state = self.select(trees, player, determinization, i)
            if not game_state.is_over():
                self.expand(trees, game_state, i)
            rewards = self.simulate(game_state)
            self.backprop(trees, rewards)

        #self.print_trees()

    def backprop(self, trees: List[base.Node], rewards):
        trees = trees.copy()
        rooted = 0
        while rooted != len(trees):
            rooted = 0
            for player in range(len(trees)):
                if trees[player] is None:
                    continue
                trees[player].visit_count += 1
                trees[player].total_reward += rewards[player]
                trees[player] = trees[player].parent

            for tree in trees:
                rooted += 1 if tree is None else 0

    def simulate(self, game_state: Game):
        rewards = []
        for player in range(len(self.trees)):
            if player == 0:
                rewards.append(self.sim_engine(player).score_boards([game_state.truth_board])[0])
            else:
                rewards.append(self.sim_engine(player).score_boards([game_state.truth_board.mirror()])[0])

        return rewards

    def expand(self, trees: List[base.Node], game_state: Game, i):
        player = None
        for i, tree in enumerate(trees):
            if isinstance(tree, base.SelfNode):
                player = i
                break

        if game_state.is_over():
            return trees, game_state

        game_state.turn = self.get_turn(player)
        sense_location = trees[player].select_child(game_state, self.sense_engine(player), i)
        trees[player] = trees[player].traverse(game_state, sense_location, self.sim_engine(player))

        trees[player].expand(game_state, self.engine_specs[player])
        move_action = trees[player].select_child(game_state, self.ucb_engine(player), i)

        for p in [player, not player]:
            game_state.turn = self.get_turn(p)
            trees[p] = trees[p].traverse(game_state, move_action, self.engine_specs[player])

        return trees

    def select(self, trees: List[base.Node], player, game_state: Game, i):
        if game_state.is_over():
            return trees, game_state
        for tree in trees:
            if len(tree.unexplored_children(game_state, self.sense_engine(player))) > 0:
                return trees, game_state

        # sense action
        sense_action = trees[player].select_child(game_state, self.sense_engine(player), i)
        trees[player] = trees[player].traverse(game_state, sense_action, self.engine_specs[player])

        # move action
        move_action = trees[player].select_child(game_state, self.ucb_engine(player), i)

        # propagate player trees to the next state
        for p in [player, not player]:
            game_state.turn = self.get_turn(p)
            trees[p] = trees[p].traverse(game_state, move_action, self.engine_specs[player])
        return self.select(trees, not player, game_state, i)

    def print_trees(self):
        names = ['Player', 'Opponent']
        for i, tree in enumerate(self.trees):
            print(f"{names[i]} Tree:")
            print("-" * 30)
            self.print_tree(tree)
            print()

    def print_tree(self, tree: base.Node, level=0):
        if tree is None:
            return
        if tree.visit_count > 0:
            prepend = '\t' * level + "+---"
            print(prepend + str(tree))
            for child in tree.children.values():
                self.print_tree(child, level=level + 1)
