# Tutorial for VPG

import gym
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from DRLagents import *


# the trunk of the net
class frontDNN(nn.Module):
    def __init__(self, inDim:int, outDim:int) -> None:
        super(frontDNN, self).__init__()
        self.layer1 = nn.Linear(inDim, 8)
        self.skiply = nn.Linear(inDim, outDim)
        self.layer2 = nn.Linear(8, outDim)

    def forward(self, x:torch.Tensor):
        y = self.skiply(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # skip connection for value-fn
        # if the value net needs to be drastically
        # different from the policy-net
        x = x + y
        return x

# create the policy network
class policyDNN(nn.Module):
    def __init__(self, inDim, num_actions, frontdnn:nn.Module):
        super(policyDNN, self).__init__()
        self.linear1 = nn.Linear(inDim, 8)
        self.linear2 = nn.Linear(8, num_actions)
        self.front = frontdnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.front(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=-1)
        return x

# create the value network
class valueDNN(nn.Module):
    def __init__(self, inDim, frontdnn:nn.Module):
        super(valueDNN, self).__init__()
        self.layer1 = nn.Linear(inDim, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 1)
        self.front = frontdnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # because value model will be updated multiple times
        # for each update of policy model, may cause chaos in
        # the shared trunk (front)
        with torch.no_grad():
            x = self.front(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def run_VPG_on_cartpole_V0(evalRender=False):
    # make a gym environment
    env = gym.make('CartPole-v0')

    # pick a suitable device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the net
    trunk = frontDNN(4, 8).to(device)
    value_model = valueDNN(8, trunk).to(device)
    policy_model = policyDNN(8, 2, trunk).to(device)

    # init necessities
    policyOptimizer = optim.Adam(policy_model.parameters(), lr=0.01)
    valueOptimizer = optim.Adam(value_model.parameters(), lr=0.01)
    trainExplortionStrategy = softMaxAction(policy_model, outputs_LogProbs=True)
    evalExplortionStrategy = greedyAction(policy_model)

    VPGagent = VPG(env, policy_model, value_model, trainExplortionStrategy, 
                    policyOptimizer, valueOptimizer, gamma=0.99, lamda=0.8, 
                    skipSteps=1, value_steps=10, MaxTrainEpisodes=400, 
                    eval_episode=5, device=device)
    trainHistory = VPGagent.trainAgent()

    # evaluate
    evalinfo = VPGagent.evaluate(evalExplortionStrategy, EvalEpisodes=5, render=evalRender)

    # close env
    env.close()

    return trainHistory, evalinfo

if __name__ == "__main__":
    trainHistory, _ = run_VPG_on_cartpole_V0(True)
    trainHistory = trainHistory['train']
    # plots the training rewards v/s episodes
    averaged_rewards = movingAverage(trainHistory['reward'])
    plt.plot(trainHistory['episode'], averaged_rewards, label="train rewards")
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.show()