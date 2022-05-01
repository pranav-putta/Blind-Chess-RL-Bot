import math
import random

import chess
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict
from enum import Enum
from game import Game
from collections import namedtuple
from util import flip_move, mirror, mirror_sense_result

eps = 1e-3
EngineSpec = namedtuple('Engine', ['sense_engine', 'sim_engine', 'ucb_engine'])


class UCBEngine(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def ucb(self, info_set, children):
        pass


class SimulationEngine(ABC):
    @abstractmethod
    def score_boards(self, boards: List[chess.Board]) -> np.ndarray:
        """
        evaluates a list of chess boards.
        :param boards: list of chess boards
        :return: returns a numpy array of scores
        """
        pass


class InformationSet(ABC):

    def __init__(self, board: Game):
        """
        constructs new information set from a board state
        :param board:
        """
        pass

    @property
    def raw(self) -> Any:
        """
        returns raw representation of information set
        :return:
        """
        pass

    @abstractmethod
    def random_sample(self) -> Game:
        """
        randomly generates a sample board state
        :return:
        """
        pass

    @abstractmethod
    def update_with_sense(self, sense_result: List[Tuple[chess.Square, chess.Piece]]):
        pass

    @abstractmethod
    def update_with_move(self, move_result):
        pass

    @abstractmethod
    def propagate_opponent_move(self, possible_moves: List[chess.Move], captured_square: chess.Piece,
                                captured_piece: bool):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def mirror(self):
        pass

    def __copy__(self):
        """
        returns a deep copy of the information set
        :return:
        """
        return self


class SenseEngine(ABC):
    @abstractmethod
    def choose_sense(self, board: InformationSet) -> chess.Square:
        """
        evaluates all sensing options given a board
        :param board: information set
        :return: returns the chosen square
        """
        pass


class Node:
    info_set: InformationSet

    children: Dict
    incoming_edge: Any
    parent: Any
    player: int

    visit_count: int
    availability_count: int
    total_reward: float

    def __init__(self, info_set: InformationSet, player, incoming_edge, parent):
        self.info_set = info_set.__copy__()
        self.children = {}
        self.incoming_edge = incoming_edge
        self.parent = parent
        self.player = player
        self.visit_count = 0
        self.total_reward = 0
        self.availability_count = 0

    def unexplored_children(self, game_state: Game, engine):
        return [move for move in game_state.get_moves() if move not in self.children]

    def select_child(self, game_state: Game, engine):
        raise NotImplementedError

    def traverse(self, game_state: Game, action, engine):
        raise NotImplementedError

    def expand(self, game_state: Game, engine_spec: EngineSpec):
        raise NotImplementedError

    @property
    def first(self):
        return list(self.children.values())[0]

    def print(self):
        print(self.info_set, '\n')

        # if self.player == chess.BLACK:
        # print(self.info_set.random_sample().truth_board.mirror())
        #  else:
        #     print(self.info_set, '\n')

    def __repr__(self):
        return f'{type(self).__name__} [ edge={self.incoming_edge}, visits={self.visit_count}, rewards={self.total_reward}, available={self.availability_count} ]'


class OpponentNode(Node):
    def __init__(self, info_set: InformationSet, player, incoming_edge=None, parent=None):
        super().__init__(info_set, player, incoming_edge, parent)

    @staticmethod
    def new(parent: Node, edge, engine: EngineSpec):
        req_move, move, captured_square = edge
        captured_piece = captured_square is not None

        new_node = OpponentNode(parent.info_set, parent.player, edge, parent)
        if move is not None:
            # TODO; make sure this is correct
            new_node.info_set.update_with_move((move, captured_piece))
        num_samples = 1
        samples = [new_node.info_set.random_sample().truth_board for i in range(num_samples)]
        avg_score = np.sum(engine.sim_engine.score_boards(samples)) / num_samples / 100
        new_node.total_reward = avg_score
        new_node.info_set.propagate_opponent_move([], captured_square,
                                                  captured_piece)
        return new_node

    def unexplored_children(self, game_state: Game, engine):
        if len(self.children) == 0:
            return ['_pass']
        else:
            return []

    def traverse(self, game_state: Game, action, engine):
        opponent_move_result = game_state.opponent_move_result()
        edge = '_pass' if opponent_move_result is None else opponent_move_result
        if edge not in self.children:
            self.children[edge] = SelfNode.new(self, edge)
        return self.children[edge]


class SelfNode(Node):
    @staticmethod
    def new(parent: OpponentNode, incoming_edge=None):
        return SelfNode(parent.info_set, parent.player, incoming_edge, parent)

    def select_child(self, game_state: Game, engine: SenseEngine):
        sense_location = engine.choose_sense(self.info_set)
        return sense_location

    def unexplored_children(self, game_state: Game, engine: SenseEngine):
        sense_location = engine.choose_sense(self.info_set)
        sense_result = game_state.handle_sense(sense_location)

        if self.player == chess.BLACK:
            sense_location = mirror(sense_location)
            sense_result = mirror_sense_result(sense_result)

        edge = (sense_location, tuple(sense_result))

        if edge not in self.children:
            return [edge]
        else:
            return []

    def traverse(self, game_state: Game, sense_location: chess.Square, engine):
        """

        :param game_state: true games tate
        :param sense_location: requested sense location with the assumption that player is white
        :return:
        """
        if self.player == chess.BLACK:
            sense_result = game_state.handle_sense(mirror(sense_location))
            # game state gives us sense result in terms of player, so flip it
            sense_result = mirror_sense_result(sense_result)
        else:
            sense_result = game_state.handle_sense(sense_location)

        edge = (sense_location, tuple(sense_result))
        if edge not in self.children:
            self.children[edge] = PlayNode.new(self, sense_location, sense_result)
        return self.children[edge]


class PlayNode(Node):
    @staticmethod
    def new(parent: SelfNode, sense_location, sense_result):
        new_node = PlayNode(parent.info_set, parent.player, incoming_edge=sense_location, parent=parent)
        new_node.info_set.update_with_sense(sense_result)
        return new_node

    def expand(self, game_state: Game, engine_spec: EngineSpec):
        moves = game_state.get_moves()
        for move in moves:
            # game state expects non-mirrored moves
            move_result = game_state.handle_move(move)
            if move_result[1] is None:  # illegal move happened, like moving pawn to the left/right and nothing there
                game_state.truth_board.pop()
                continue
            edge = move_result[0:3]
            if self.player == chess.BLACK:
                edge = flip_move(edge[0]), flip_move(edge[1]), mirror(edge[2])
            key = edge[1]
            if key not in self.children:
                self.children[key] = OpponentNode.new(self, edge, engine_spec)
            game_state.truth_board.pop()

    def select_child(self, game_state: Game, engine: UCBEngine):
        available_moves = game_state.get_moves()

        if self.player == chess.BLACK:
            for i, move in enumerate(available_moves):
                available_moves[i] = flip_move(move)
        if len(available_moves) == 1:
            return available_moves[0]
        moves = []
        for move in available_moves:
            if move in self.children:
                self.children[move].availability_count += 1
                moves.append(move)

        children = []
        for move in moves:
            children.append(self.children[move])
        ucbs = sorted(zip(engine.ucb(self.info_set, children), moves), key=lambda item: item[0], reverse=True)
        if len(ucbs) == 0:
            return None
        return ucbs[0][1]

    def traverse(self, game_state: Game, move, engine):
        if move is None:
            return self
        # convert move to proper game_state
        if self.player == chess.BLACK:
            move = flip_move(move)
        move_result = game_state.handle_move(move)
        edge = move_result[0:3]
        if self.player == chess.BLACK:
            edge = flip_move(edge[0]), flip_move(edge[1]), mirror(edge[2])
        key = edge[1]
        if key not in self.children:
            self.children[key] = OpponentNode.new(self, edge, engine)
        return self.children[key]


class PolicyEngine(ABC):

    @abstractmethod
    def generate_policy(self, self_info_set: InformationSet) -> np.ndarray:
        """
        generates a policy for a given state
        :return:
        """
        pass

    def generate_policies(self, self_info_sets: List[InformationSet]) -> np.ndarray:
        policies = []
        for info_set in self_info_sets:
            policies.append(self.generate_policy(info_set))
        return np.arary(policies)

    def prob_given_action(self, policy, move):
        return policy[move]
