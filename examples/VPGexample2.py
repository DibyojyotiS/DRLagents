# Tutorial for VPG

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

# the trunk of the net
class frontDNN(nn.Module):
    def __init__(self, inDim:int, outDim:int, hDims:'list[int]', activation = F.relu) -> None:
        super(frontDNN, self).__init__()
        self.inputlayer = nn.Linear(inDim, hDims[0])
        self.hiddenlayers = nn.ModuleList([nn.Linear(hDims[i], hDims[i+1]) for i in range(len(hDims)-1)])
        self.outputlayer = nn.Linear(hDims[-1], outDim)
        self.activation = activation

    def forward(self, x:torch.Tensor):
        t = self.activation(self.inputlayer(x))
        for layer in self.hiddenlayers:
            t = self.activation(layer(t))
        t = self.activation(self.outputlayer(t))
        return t

# create the policy network
class policyDNN(nn.Module):
    def __init__(self, inDim, num_actions, frontdnn:nn.Module):
        super(policyDNN, self).__init__()
        self.linear1 = nn.Linear(inDim, num_actions)
        self.front = frontdnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.front(x)
        x = self.linear1(x)
        x = F.log_softmax(x, dim=-1)
        return x

# create the value network
class valueDNN(nn.Module):
    def __init__(self, inDim, frontdnn:nn.Module):
        super(valueDNN, self).__init__()
        self.layer1 = nn.Linear(inDim, 1)
        self.front = frontdnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # because value model will be updated multiple times
        # for each update of policy model
        with torch.no_grad():
            # set the existing gradients of the front model to zeros
            # otherwise optimizer will update front as well
            self.front.zero_grad()
            x = self.front(x)
        x = self.layer1(x)
        return x

# create the net
trunk = frontDNN(4, 8, [8]).to(device)
value_model = valueDNN(8, trunk).to(device)
policy_model = policyDNN(8, 2, trunk).to(device)

# init necessities
policyOptimizer = optim.Adam(policy_model.parameters(), lr=0.01)
valueOptimizer = optim.Adam(value_model.parameters(), lr=0.01)
trainExplortionStrategy = softMaxAction(policy_model, outputs_LogProbs=True)
evalExplortionStrategy = greedyAction(policy_model)

VPGagent = VPG(env, policy_model, value_model, trainExplortionStrategy, policyOptimizer, 
                valueOptimizer, gamma=0.99, skipSteps=1, MaxTrainEpisodes=400, device=device)
trainHistory = VPGagent.trainAgent()

# render
VPGagent.evaluate(evalExplortionStrategy, EvalEpisodes=5, render=True)

# plots the training rewards v/s episodes
averaged_rewards = movingAverage(trainHistory['trainRewards'])
plt.plot([*range(len(trainHistory['trainRewards']))], averaged_rewards, label="train rewards")
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()