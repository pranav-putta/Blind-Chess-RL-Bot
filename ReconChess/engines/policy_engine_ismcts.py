import chess

import ReconChess.engines.base as base
import numpy as np
from typing import List, Tuple, Any, Generic, Type
from enum import Enum




class BasicNode:
    info_set: base.InformationSet
    children: List
    parent: Any
    player: int


class Node:
    info_set: base.InformationSet
    truth: base.Game
    children: List
    determinization: chess.Board
    value: float
    parent: Any
    player: int

    def __init__(self, truth, parent, info_set: base.InformationSet, determinization=None):
        self.truth = truth
        self.info_set = info_set
        self.children = []
        self.determinization = determinization
        self.parent = parent

    def find_or_create_child(self, node) -> Any:
        # TODO: implement find_or_create_child
        return self

    def update_move(self, move: chess.Move):
        result = self.truth.handle_move(move)
        self.determinization = None
        self.info_set.update_with_move(result)

    @property
    def terminal(self):
        return self.determinization.is_game_over()

    @property
    def possible_moves(self):
        return self.truth.board.generate_pseudo_legal_moves()

    def __copy__(self):
        node = Node(self.truth.__copy__(), self.parent, info_set=self.info_set.__copy__(),
                    determinization=self.determinization.copy())
        return node


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
        self.p1_tree = None
        self.p2_tree = None
        self.num_iters = num_iterations
        self.reset_info_set_on_determinization = reset_info_set_on_determinization

    def generate_policy(self, player_info_set: base.InformationSet, other_info_set: base.InformationSet) -> np.ndarray:
        # TODO: fix monte carlo call
        return self.mcts(player_info_set, other_info_set, None)

    def ucb(self, move):
        return 0

    def determinization(self, state: Node):
        """
        determinizes a provided state.
        samples a board from the information set and creates a new node.
        :param state:
        :return:
        """
        determinization = state.info_set.random_sample()
        if self.reset_info_set_on_determinization:
            return Node(state.truth, state, determinization=determinization,
                        info_set=self.info_set_type(base.Game.empty()))
        else:
            return Node(state.truth, state, determinization=determinization, info_set=state.info_set)

    def mcts(self, p1_info_set: base.InformationSet, p2_info_set: base.InformationSet,
             game: base.Game) -> np.ndarray:

        p1_tree = Node(game, None, p1_info_set)
        p2_tree = Node(game, None, p2_info_set)

        player = 0

        for i in range(self.num_iters):
            # perform initial determinization
            trees = self.select([p1_tree, p2_tree], player)
            if not trees[player].terminal:
                self.expand(trees)
            self.simulate(trees)

    def has_visited(self, node):
        # TODO: fix this
        return False

    def select_move(self, node: Node):
        best_ucb, best_move = float('-inf'), None
        for next_board in node.children:
            value = self.ucb(next_board)
            if value > best_ucb:
                best_move = next_board
                best_ucb = value
        return best_move

    def backprop(self, trees: List[Node], reward):
        all_rooted = True
        for tree in trees:
            if tree is not None:
                all_rooted = False
                break
        if all_rooted:
            return

        for i, tree in enumerate(trees):
            if tree is None:
                continue
            tree.value += reward
            trees[i] = tree.parent

    def simulate(self, trees: List[Node]):
        boards = []
        for tree in trees:
            if tree.determinization is None:
                print("Error: tried to simulate on non-determinized node")
                exit()
            boards.append(tree.determinization)
        results = self.eval_engine.evaluate_rewards(boards)
        for i, tree in enumerate(trees):
            tree.value = results[i]

    def expand(self, trees: List[Node]):
        for i, node in enumerate(trees):
            for move in node.possible_moves:
                tmp = node.__copy__()
                tmp.update_move(move)
                node.find_or_create_child(node)

    def select(self, trees: List[Node], player: int):
        p_tree: Node = trees[player]
        # perform sense
        sq = self.sense_engine.choose_sense(p_tree.info_set)  # get the sense
        result = p_tree.truth.handle_sense(sq)
        p_tree.info_set.update_with_sense(result)

        # perform determinization
        tmp = self.determinization(p_tree)
        p_tree = p_tree.find_or_create_child(tmp)

        while not p_tree.terminal and not self.has_visited(p_tree):
            # calculate best move
            move = self.select_move(p_tree)
            # traverse down all trees for this move
            for i, tree in enumerate(trees):
                trees[i] = tree.find_or_create_child(move)
            p_tree = trees[player]
            # propagate opponent move
            p_tree.info_set.propagate_opponent_move()

            # perform sense
            sq = self.sense_engine.choose_sense(p_tree.info_set)  # get the sense
            result = p_tree.truth.handle_sense(sq)
            p_tree.info_set.update_with_sense(result)

            # perform determinization
            tmp = self.determinization(p_tree)
            p_tree = p_tree.find_or_create_child(tmp)
        return trees
