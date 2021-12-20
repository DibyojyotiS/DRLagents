# Reinforce with baseline (VPG) is implemented here 
# This particular version also uses entropy in the policy loss

from time import perf_counter
from typing import Union
import gym
import torch
from explorationStrategies import Strategy
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

class VPG:

    def __init__(self, 
                # training necessities
                env:gym.Env, policy_model: nn.Module, value_model:nn.Module,
                trainExplortionStrategy: Strategy,
                policy_optimizer: Optimizer, 
                value_optimizer: Optimizer, 

                # optional training necessities
                make_state = lambda listOfObs, listOfInfos: listOfObs[-1],
                gamma = 0.8,
                beta = 0.1,
                MaxTrainEpisodes = 500,
                MaxStepsPerEpisode = None,
                value_steps = 10,

                # miscellaneous
                skipSteps = 0,
                breakAtReward = float('inf'),
                printFreq = 50,
                device= torch.device('cpu')) -> None:
        """ policy model maps the states to action log-probablities \n
        # NOTE: send log-probablities!!! """

        # training necessities
        self.env = env
        self.policy_model = policy_model
        self.value_optimizer = value_optimizer
        self.value_model = value_model
        self.trainExplortionStrategy = trainExplortionStrategy
        self.policy_optimizer = policy_optimizer

        # optional training necessities
        self.make_state = make_state
        self.gamma = gamma
        self.beta  = beta
        self.MaxTrainEpisodes = MaxTrainEpisodes
        self.MaxStepsPerEpisode = MaxStepsPerEpisode
        self.value_steps = value_steps

        # miscellaneous
        self.skipSteps = skipSteps
        self.breakAtReward = breakAtReward
        self.printFreq = printFreq
        self.device = device

        # required inits
        self._initBookKeeping()


    def trainAgent(self):
        """ The main function to train the policy model """
        
        timeStart = perf_counter()
        for episode in range(self.MaxTrainEpisodes):
            # get the trajectory
            trajectory, totalReward, total_steps = self._genetate_trajectory()
            # optimize models
            policyloss, valueloss = self._optimizeAgent(trajectory)
            # do book-keeping
            self._performBookKeeping(totalReward, total_steps, policyloss, valueloss, perf_counter()-timeStart())
            # show progress output
            if episode % self.printFreq == 0:
                print(f'episode: {episode} -> reward: {totalReward} time-elasped: {perf_counter()-timeStart:.2f}s')
            # early breaking
            if totalReward >= self.breakAtReward:
                print(f'stopping at episode {episode}')
                break
        # print the last episode if not already printed
        if episode % self.printFreq != 0:
            print(f'episode: {episode} -> reward: {totalReward} time-elasped: {perf_counter()-timeStart:.2f}s')

        # just cuirious to know the total time spent...
        print("total time elasped:", perf_counter() - timeStart,'s')    
        
        return {
            'trainRewards': self.trainRewards, 
            'episodeSteps': self.episodeSteps, 
            'policylosses': self.policylosses, 
            'valuelosses': self.valuelosses,
            'wallTime': self.wallTime
        }


    def _optimizeAgent(self, trajectory:dict):
        """ to optimize the agent """
        # compute the partial return at every step
        partial_returns = self._compute_Gt(trajectory['reward'])

        # action_probablities
        action_logprobs = trajectory['log_prob']
        entropies = trajectory['entropy']

        # baseline
        values = self.value_model(trajectory['state'])

        # compute policy-loss
        action_advantages = partial_returns - values.detach()
        policyLoss = - (action_advantages*action_logprobs \
                            + self.beta*entropies).mean()

        # compute value-loss
        valueLoss = F.mse_loss(values, partial_returns)

        # grad-step policy model
        self.policy_optimizer.zero_grad()
        policyLoss.backward()
        for param in self.policy_model.parameters(): param.grad.clamp_(-1,1)
        self.policy_optimizer.step()

        # grad-step value model
        for value_step in range(self.value_steps):
            self.value_optimizer.zero_grad()
            valueLoss.backward()
            for param in self.value_model.parameters(): param.grad.clamp_(-1,1)
            self.value_optimizer.step()

        return policyLoss.item(), valueLoss.item()


    def _genetate_trajectory(self):
        """ runs an episode and stores the history as a dict with the 
        keys 'state', 'action', 'reward'.
        returns: trajectory_dict, total_reward, total_steps
        NOTE: the states also include the terminal state, making
                its length 1 more than actions, rewards and 
                partial returns """
        trajectory = {'state':[], 'action':[], 'reward':[], 'log_prob':[], 'entropy':[]}
        # bookeeping counters
        total_reward = 0.0
        total_steps = 0

        done = False
        observation = self.env.reset()
        # user defines this func to make a state from a list of observations and infos
        state = self.make_state([observation for _ in range(self.skipSteps)], 
                                [None for _ in range(self.skipSteps)])
        while not done:
            # take the action and handle frame skipping
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            action, log_prob, entropy = self.trainExplortionStrategy.select_action(state_tensor, grad=True, logProb_n_entropy=True)
            nextState, accumulatedReward, sumReward, done, stepsTaken = self._take_steps(action.item())
            # append to trajectory
            trajectory['state'].append(state)
            trajectory['action'].append(action)
            trajectory['log_prob'].append(log_prob) # log-probabliy of taken action
            trajectory['entropy'].append(entropy) # entropy of action probablities
            trajectory['reward'].append(accumulatedReward)
            # update counters
            total_reward+=sumReward
            total_steps+=stepsTaken
            # update state
            state = nextState
        # append the terminal state
        trajectory['state'].append(state)
        # convert the trajectory to a tensor
        trajectory = self._astensor(trajectory)
        return trajectory, total_reward, total_steps


    def _compute_Gt(self, rewards:Union[list, torch.Tensor]):
        ''' computes the partial returns at each time-step '''
        total_steps = len(rewards)
        partial_Gts = [0.0]*total_steps 
        prev_Gt = 0 # partial return from i+1 onwards
        for i in range(total_steps-1,-1,-1):
            ith_Gt = rewards[i] + self.gamma*prev_Gt # Gt from i onwards
            partial_Gts[i] = ith_Gt
            prev_Gt = ith_Gt
        partial_Gts = torch.tensor(partial_Gts, dtype=torch.float32, device=self.device)
        return partial_Gts


    def _take_steps(self, action, render=False):
        # also handles the frame skipping and rendering for evaluation

        accumulatedReward = 0 # the reward the model will see
        sumReward = 0 # to keep track of the total reward in the episode
        stepsTaken = 0 # to keep track of the total steps in the episode

        observationList = []
        infoList = []

        for skipped_step in range(self.skipSteps):

            # repeate the action
            nextObservation, reward, done, info = self.env.step(action.item())

            if render: self.env.render()

            accumulatedReward += reward # reward * self.gamma**skipped_step
            sumReward += reward
            stepsTaken += 1

            observationList.append(nextObservation)
            infoList.append(info)

            # if done pad the lists with the latest information
            if done:
                padLen = self.skipSteps-len(observationList)
                observationList.extend([nextObservation for _ in range(padLen)])        
                infoList.extend([info for _ in range(padLen)])
                break # no more steps to skip
        
        # compute the next state 
        nextState = self.make_state(observationList, infoList)

        return nextState, accumulatedReward, sumReward, done, stepsTaken


    def _initBookKeeping(self):
        self.trainRewards = []
        self.episodeSteps = []
        self.policylosses = []
        self.valuelosses = []
        self.wallTime = []


    def _performBookKeeping(self, trainReward, steps, policyloss, valueloss, timeElasped):
        self.trainRewards.append(trainReward)
        self.episodeSteps.append(steps)
        self.policylosses.append(policyloss)
        self.valuelosses.append(valueloss)
        self.wallTime.append(timeElasped)   

    
    def _astensor(self, args:'dict[str, list]') -> 'dict[str, torch.Tensor]':
        for k in args.keys():
            args[k] = torch.tensor(args[k], dtype=torch.float32, device=self.device)
        return args