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

# create the value network
class valueDNN(nn.Module):
    def __init__(self, inDim, outDim, hDim, activation = F.relu):
        super(valueDNN, self).__init__()
        self.inputlayer = nn.Linear(inDim, hDim[0])
        self.hiddenlayers = nn.ModuleList([nn.Linear(hDim[i], hDim[i+1]) for i in range(len(hDim)-1)])
        self.valuelayer = nn.Linear(hDim[-1], 1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.inputlayer(x))
        for layer in self.hiddenlayers:
            x = self.activation(layer(x))
        value = self.valuelayer(x)
        return value

# create the policy network
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
        t = F.log_softmax(t, -1)
        return t


def run_VPG_on_cartpole_V0(evalRender=False):
    # make a gym environment
    env = gym.make('CartPole-v0')

    # pick a suitable device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the net
    value_model = valueDNN(inDim=4, outDim=1, hDim=[8,8], activation=F.relu).to(device)
    policy_model = net(inDim=4, outDim=2, hDim=[8,8], activation=F.relu).to(device)

    # init necessities
    policyOptimizer = optim.Adam(policy_model.parameters(), lr=0.01)
    valueOptimizer = optim.Adam(value_model.parameters(), lr=0.01)
    trainExplortionStrategy = softMaxAction(policy_model, outputs_LogProbs=True)
    evalExplortionStrategy = greedyAction(policy_model)

    VPGagent = VPG(env, policy_model, value_model, trainExplortionStrategy, 
                    policyOptimizer, valueOptimizer, gamma=0.99, lamda=0.8,
                    skipSteps=1, value_steps=10, MaxTrainEpisodes=400, device=device)
    trainHistory = VPGagent.trainAgent()

    # evaluate
    evalinfo = VPGagent.evaluate(evalExplortionStrategy, EvalEpisodes=5, render=evalRender)

    # close env
    env.close()

    return trainHistory, evalinfo


if __name__ == "__main__":
    trainHistory, _ = run_VPG_on_cartpole_V0(True)
    # plots the training rewards v/s episodes
    averaged_rewards = movingAverage(trainHistory['trainRewards'])
    plt.plot([*range(len(trainHistory['trainRewards']))], averaged_rewards, label="train rewards")
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.show()