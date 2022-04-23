from engines import StockFishNNEvalEngine, VanillaISMCTSPolicyEngine, RandomSenseEngine


def self_play_games(n=10):
    samples = []
    for i in range(n):
        # play game and return samples
        plays = []
        samples.append(plays)
        pass
    return samples


def train():
    policy_network = None

    for iterations in range(100):
        # generate samples
        samples = self_play_games()

        # run gradient descent on samples

        # play updated network against old network
        # if updated network beats old one more than 50% of the time, update policy network
