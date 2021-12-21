# Tutorial for both DDDQN and DDDQN with Prioritized Experience Replay
# for prioritized replay version uncomment line 53

import gym
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from DRLagents import *


# make a gym environment
env = gym.make('CartPole-v0')


# pick a suitable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create the deep network
class duellingDNN(nn.Module):
    def __init__(self, inDim, outDim, hDim, activation = F.relu):
        super(duellingDNN, self).__init__()
        self.inputlayer = nn.Linear(inDim, hDim[0])
        self.hiddenlayers = nn.ModuleList([nn.Linear(hDim[i], hDim[i+1]) for i in range(len(hDim)-1)])
        self.valuelayer = nn.Linear(hDim[-1], 1)
        self.actionadv = nn.Linear(hDim[-1], outDim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.inputlayer(x))
        for layer in self.hiddenlayers:
            x = self.activation(layer(x))
        advantages = self.actionadv(x)
        values = self.valuelayer(x)
        qvals = values + (advantages - advantages.mean())
        return qvals


# init necessities
duellingQnetwork = duellingDNN(inDim=4, outDim=2, hDim=[8,8], activation=F.relu).to(device)
optimizer = optim.Adam(duellingQnetwork.parameters(), lr=0.001)
trainExplortionStrategy = epsilonGreedyAction(duellingQnetwork, 0.5, 0.05, 100)
evalExplortionStrategy = greedyAction(duellingQnetwork)


## choose wether prioritized replay buffer or uniform sampling replay buffer or implement your own
replayBuffer = ExperienceReplayBuffer(bufferSize=10000) # uniform sampling, windowed memory
# replayBuffer = PrioritizedExperienceRelpayBuffer(bufferSize=10000, alpha=0.6, beta=0.2, beta_rate=0.005) # prioritized sampling


# define the training strategy DQN in our example
D3QNagent = DDQN(env, duellingQnetwork, trainExplortionStrategy, optimizer, replayBuffer, 64,
                MaxTrainEpisodes=250, skipSteps=1, device=device, polyak_average=True, update_freq=5)


# train the model
trainHistory = D3QNagent.trainAgent()


# evaluate the model
evalRewards = D3QNagent.evaluate(evalExplortionStrategy, EvalEpisodes=5, render=True)


# plots the training rewards v/s episodes
averaged_rewards = movingAverage(trainHistory['trainRewards'])
plt.plot([*range(len(trainHistory['trainRewards']))], averaged_rewards, label="train rewards")
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()