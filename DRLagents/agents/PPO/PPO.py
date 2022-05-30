import torch

from DRLagents.explorationStrategies import Strategy

# need a replay buffer for ppo

class PPO:
    def __init__(self) -> None:
        pass

    def trainAgent(self):
        """train the actor-critic network"""
        pass

    def _parallel_trajectories(self, N):
        """ returns N trajectories computed in parallel """
        pass

    def _optimize(self):
        pass

    def evaluate(self, evalExplortionStrategy:Strategy, EvalEpisodes=1, render=False, verbose=True):
        """ Evaluate the model for EvalEpisodes number of episodes """
        pass

