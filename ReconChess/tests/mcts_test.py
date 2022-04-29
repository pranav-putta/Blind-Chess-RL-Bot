import chess

from engines import ISMCTSPolicyEngine, RandomSenseEngine, RandomEvaluationEngine, PiecewiseInformationSet
from game import Game

sense_engine = RandomSenseEngine()
eval_engine = RandomEvaluationEngine()
policy_engine = ISMCTSPolicyEngine(sense_engine, eval_engine, PiecewiseInformationSet, 10)

game = Game()
info_set = PiecewiseInformationSet(game)
info_set2 = PiecewiseInformationSet(game)
policy_engine.generate_policy(info_set, info_set2)
