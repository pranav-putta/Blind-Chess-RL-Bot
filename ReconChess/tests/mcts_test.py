import chess

from ReconChess.engines import ISMCTSPolicyEngine, RandomSenseEngine, RandomEvaluationEngine, PiecewiseInformationSet, \
    Game

sense_engine = RandomSenseEngine()
eval_engine = RandomEvaluationEngine()
policy_engine = ISMCTSPolicyEngine(sense_engine, eval_engine, PiecewiseInformationSet, 10)

game = Game(chess.Board())
info_set = PiecewiseInformationSet(game)
policy_engine.generate_policy(info_set, game)
