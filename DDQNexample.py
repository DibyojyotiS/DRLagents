# Tutorial for both DDQN and DDQN with Prioritized Experience Replay
# for prioritized replay version uncomment line 48

import gym
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim

from algorithms import DDQN
from replaybuffers import ExperienceReplayBuffer, PrioritizedExperienceRelpayBuffer
from explorationStrategies import epsilonGreedyAction
from explorationStrategies.greedyStrategy import greedyAction
from utils import movingAverage

# make a gym environment
env = gym.make('CartPole-v0')

# pick a suitable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create the deep network
class net(nn.Module):
    def __init__(self, inDim, outDim, hDim, activation = F.relu):
        super(net, self).__init__()
        self.inputlayer = nn.Linear(inDim, hDim[0])
        self.hiddenlayers = nn.ModuleList([nn.Linear(hDim[i], hDim[i+1]) for i in range(len(hDim)-1)])
        self.outputlayer = nn.Linear(hDim[-1], outDim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.activation(self.inputlayer(x))
        for layer in self.hiddenlayers:
            t = self.activation(layer(t))
        t = self.outputlayer(t)
        return t


# init necessities
Qnetwork = net(inDim=4, outDim=2, hDim=[8,8], activation=F.relu).to(device)
optimizer = optim.Adam(Qnetwork.parameters(), lr=0.001)
trainExplortionStrategy = epsilonGreedyAction(Qnetwork, 0.5, 0.1, 100)
evalExplortionStrategy = greedyAction(Qnetwork)

## choose wether prioritized replay buffer or uniform sampling replay buffer or implement your own
replayBuffer = ExperienceReplayBuffer(bufferSize=10000) # for uniform sampling
# replayBuffer = PrioritizedExperienceRelpayBuffer(bufferSize=10000, alpha=0.6, beta=0.2, beta_rate=0.002) # for prioritized sampl..


# define the training strategy DQN in our example
DQNagent = DDQN(Qnetwork, env, trainExplortionStrategy, optimizer, replayBuffer, 64, 
                MaxTrainEpisodes=500, skipSteps=0, polyak_average=True, device=device)
                # might want to do MaxTrainEpisodes=250 for prioritized buffer


# train the model
trainRewards, steps, trainloss, wallTime = DQNagent.trainAgent()


# evaluate the model
evalRewards = DQNagent.evaluate(evalExplortionStrategy, EvalEpisodes=5, render=True)


# plots the training rewards v/s episodes
averaged_rewards = movingAverage(trainRewards)
plt.plot([*range(len(trainRewards))], averaged_rewards, label="train rewards")
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()