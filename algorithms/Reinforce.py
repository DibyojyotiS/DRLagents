# Reinforce with baseline is implemented here 
# with an option to not include the baseline

import gym
import torch
from explorationStrategies import Strategy
from torch import nn, optim


class Reinforce:

    def __init__(self, 
                # training necessities
                policy_model: nn.Module, env:gym.Env,
                trainExplortionStrategy: Strategy,
                optimizer: optim.Optimizer, 

                # optional training necessities
                MaxTrainEpisodes = 500,
                make_state = lambda listOfObs, listOfInfos: listOfObs[-1],

                # miscellaneous
                skipSteps = 0,
                breakAtReward = float('inf'),
                printFreq = 50,
                device= torch.device('cpu')) -> None:
        """ policy model maps the states to action probablities """

        # training necessities
        self.policy_model = policy_model
        self.trainExplortionStrategy = trainExplortionStrategy
        self.optimizer = optimizer

        # optional training necessities
        self.MaxTrainEpisodes = MaxTrainEpisodes
        self.make_state = make_state

        # miscellaneous
        self.skipSteps = skipSteps
        self.breakAtReward = breakAtReward
        self.printFreq = printFreq
        self.device = device

        # required inits
        self._initBookKeeping()


    def trainAgent(self):
        """ The main function to train the policy model """
        
        for episode in range(self.MaxTrainEpisodes):
            # get the trajectory
            # for each step in trajectory accumulate the gradient
            trajectory, total_reward, total_steps = self._genetate_trajectory()
            


    def _genetate_trajectory(self):
        """ runs an episode and returns the entire history as
        a dict with keys 'state', 'action', 'reward', 'nextState' """
        trajectory = {'state':[], 'reward':[], 'action':[], 'nextState':[]}
        # bookeeping counters
        total_reward = 0
        total_steps = 0

        done = False
        info = None
        observation = self.env.reset()
        # user defines this func to make a state from a list of observations and infos
        state = self.make_state([observation for _ in range(self.skipSteps)], 
                                [info for _ in range(self.skipSteps)])
        while not done:
            # take the action and handle frame skipping
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            action = self.trainExplortionStrategy.select_action(state_tensor).item()
            nextState, accumulatedReward, sumReward, done, stepsTaken = self._take_steps(action)
            # append to trajectory
            trajectory['state'].append(state)
            trajectory['action'].append(action)
            trajectory['reward'].append(accumulatedReward)
            trajectory['nextState'].append(nextState)
            state = nextState
            # update counters
            total_reward+=sumReward
            total_steps+=stepsTaken
        return trajectory, total_reward, total_steps


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
        self.loss = []
        self.wallTime = []


    def _performBookKeeping(self, trainReward, steps, totalLoss, timeElasped):
        self.trainRewards.append(trainReward)
        self.episodeSteps.append(steps)
        self.loss.append(totalLoss)
        self.wallTime.append(timeElasped)   