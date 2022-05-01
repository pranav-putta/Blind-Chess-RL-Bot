import chess

from engines import ISMCTSPolicyEngine, PiecewiseSenseEngine, StockFishEvaluationEngine, PiecewiseInformationSet
from engines.game import Game
import engines.base as base

sense_engine = PiecewiseSenseEngine()
eval_engine = StockFishEvaluationEngine(depth=10)
engine_spec = base.EngineSpec(sense_engine, eval_engine)
policy_engine = ISMCTSPolicyEngine(engine_spec, num_iters=100)

game = Game()
info_set = PiecewiseInformationSet(game)
info_set2 = PiecewiseInformationSet(game)
policy_engine.generate_policy(info_set)
eval_engine.close()
