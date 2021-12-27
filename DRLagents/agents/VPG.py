# Reinforce with baseline (VPG) is implemented here 
# This particular version also uses entropy in the policy loss

from time import perf_counter
from typing import Union

import gym
import torch
import torch.nn.functional as F
from DRLagents.agents.helper_funcs.helper_funcs import clip_grads
from DRLagents.explorationStrategies import Strategy
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer


class VPG:
    ''' ## Vanila Policy Gradient with Baseline

    The baseline used is the state-value function.
    The value is substracted from the return at any 
    time step giving the action advantage. 
    Entropy is also added to the VPG loss after 
    being scaled by a factor of beta. '''

    def __init__(self, 
                # training necessities
                env:gym.Env, policy_model: nn.Module, value_model:nn.Module,
                trainExplortionStrategy: Strategy,
                policy_optimizer: Optimizer, 
                value_optimizer: Optimizer, 

                # optional training necessities
                make_state = lambda listOfObs, listOfInfos: listOfObs[-1],
                gamma = 0.8,
                beta = 0.3,
                MaxTrainEpisodes = 500,
                MaxStepsPerEpisode = None,
                trajectory_seg_length = None,
                value_steps = 10,

                # miscellaneous
                skipSteps = 0,
                breakAtReward = float('inf'),
                printFreq = 50,
                device= torch.device('cpu')) -> None:
        """ 
        ### pls read the notes at the bottom

        ## training necessities

        env: a gym env, that behaves similar to standard gym environments. Currently tested on env like cartpole.

        policy_model: nn.Module that maps states to log-probablities

        value_model: nn.Module that maps states to state-Values

        trainExplortionStrategy: training strategy similar to the classes defined in explorationStrategies.

        policy_optimizer: any optimizer with same behaviour one from torch.optim

        value_optimizer: any optimizer with same behaviour one from torch.optim

        ## optional training necessities

        make_state: function that takes in list of observatons and infos from
                    env.step to make a state. This function should handle info
                    as list of Nones, since gym.env.reset doesnot return info.

        gamma: the discount factor

        beta: entropy weight

        MaxTrainEpisodes: maximum number of episodes to train for

        MaxStepsPerEpisode: break the episode if number of steps taken reaches or exceeds this

        trajectory_seg_length: if not None then the original trajectory is 
                                broken up into non-overlapping segments and 
                                each segment is treated as a trajectory. The 
                                segment length is <= trajectory_seg_length. 
                                But note that the computed returns will not 
                                be accurate. It is preferable to keep this as 
                                None or as large as possible.

        value_steps: the number of gradient steps for value-model for every 
                    gradient step for the policy model.

        ## miscellaneous 

        skipSteps: the number of steps to repeate the previous action (model not optimized at skipped steps)
        
        breakAtReward: break training when the reward reaches this number
        
        printFreq: print episode reward every printFreq episode

        # Implementation notes:\n
        NOTE: It is assumed that optimizer is already setup with the network parameters and learning rate.
        NOTE: The make_state function gets an input of a list of observations and infos corresponding to the skipped and the current states.
        NOTE: The initial list of observations is the initial observation repeated skipSteps+1 number of times.
        NOTE: If episode terminates while skipping, the list is padded with the last observation and info
        NOTE: The models are not updated within the skipSteps steps.
        NOTE: policy_model maps the states to action log-probablities
        NOTE: value_model returns a single value for each state """

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
        self.trajectory_seg_length = trajectory_seg_length
        self.value_steps = value_steps

        # miscellaneous
        self.skipSteps = skipSteps
        self.breakAtReward = breakAtReward
        self.printFreq = printFreq
        self.device = device

        # required inits
        self._initBookKeeping()


    def trainAgent(self, render=False):
        """ The main function to train the policy and value models """
        
        timeStart = perf_counter()

        for episode in range(self.MaxTrainEpisodes):

            # get the trajectory segments & optimize models
            # trajectory, totalReward, total_steps = self._genetate_trajectory()
            for output_tupple in self._genetate_trajectory(render):
                trajectory, totalReward, total_steps = output_tupple
                policyloss, valueloss, avgEntropy = self._optimizeAgent(trajectory)

            # do book-keeping
            self._performBookKeeping(totalReward, total_steps, policyloss, 
                                        valueloss, perf_counter()-timeStart, avgEntropy)

            # show progress output
            if episode % self.printFreq == 0:
                print(f'episode: {episode} -> reward: {totalReward}, steps:{total_steps} time-elasped: {perf_counter()-timeStart:.2f}s')

            # early breaking
            if totalReward >= self.breakAtReward:
                print(f'stopping at episode {episode}')
                break

        # print the last episode if not already printed
        if episode % self.printFreq != 0:
            print(f'episode: {episode} -> reward: {totalReward}, steps:{total_steps} time-elasped: {perf_counter()-timeStart:.2f}s')

        # just cuirious to know the total time spent...
        print("total time elasped:", perf_counter() - timeStart,'s')    
        
        return {
            'trainRewards': self.trainRewards, 
            'episodeSteps': self.episodeSteps, 
            'policylosses': self.policylosses, 
            'valuelosses': self.valuelosses,
            'wallTime': self.wallTime,
            'entropies':self.entropies
        }


    def _optimizeAgent(self, trajectory:'dict[str, Tensor]'):
        """ to optimize the agent """
        # compute the partial return at every step
        partial_returns = self._compute_Gt(trajectory['reward'])

        # action_probablities
        action_logprobs = trajectory['log_prob']
        mean_entropy = trajectory['entropy'].mean()

        # baseline
        values = self.value_model(trajectory['state'][:-1]).squeeze(dim=-1)

        # compute policy-loss
        action_advantages = partial_returns - values.detach()
        policyLoss = - (action_advantages*action_logprobs).mean() \
                            + self.beta*mean_entropy

        # grad-step policy model
        self.policy_optimizer.zero_grad()
        policyLoss.backward()
        clip_grads(self.policy_model)
        self.policy_optimizer.step()

        # grad-step value model
        valueLoss = self._value_model_grad_step(values, partial_returns)
        for value_step in range(1, min(self.value_steps, values.shape[0])):
            values = self.value_model(trajectory['state'][:-1]).squeeze()
            valueLoss = self._value_model_grad_step(values, partial_returns)

        return policyLoss.item(), valueLoss, mean_entropy.item()


    def _genetate_trajectory(self, render=False):
        """ 
        This is a generator function to generate trajectories

        It runs an episode and stores the history as a dict with the 
        keys 'state', 'action', 'reward', 'log_prob', 'entropy'
        returns: trajectory_dict, total_reward, total_steps
        NOTE: the states also include the terminal state, making
                its length 1 more than actions, rewards and 
                partial returns """
        
        trajectory = {'state':[], 'action':[], 'reward':[], 'log_prob':[], 'entropy':[]}

        # bookeeping counters
        total_reward = 0.0
        total_steps = 0
        total_appends = 0 # track length of original trajectory

        done = False
        observation = self.env.reset()

        # render
        if render: self.env.render()

        # user defines this func to make a state from a list of observations and infos
        state = self.make_state([observation for _ in range(self.skipSteps)], 
                                [None for _ in range(self.skipSteps)])
        
        while not done:
            # take the action and handle frame skipping
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            action, log_prob, entropy = self.trainExplortionStrategy.select_action(state_tensor, grad=True, logProb_n_entropy=True)
            nextState, accumulatedReward, sumReward, done, stepsTaken = self._take_steps(action)
            # append to trajectory
            trajectory['state'].append(state)
            trajectory['action'].append(action)
            trajectory['log_prob'].append(log_prob) # log-probabliy of taken action
            trajectory['entropy'].append(entropy) # entropy of action probablities
            trajectory['reward'].append(accumulatedReward)
            # update counters
            total_reward+=sumReward
            total_steps+=stepsTaken
            total_appends+=1
            # update state
            state = nextState

            # render
            if render: self.env.render()

            # break episode is self.MaxStepsPerEpisode is exceeded
            if self.MaxStepsPerEpisode and total_steps >= self.MaxStepsPerEpisode: break

            # yeild turncated trajectory if self.trajectory_seg_length is not None
            if self.trajectory_seg_length and total_appends%self.trajectory_seg_length == 0:
                trajectory['state'].append(state) # proxy for terminal state
                trajectory = self._trajectory_to_tensor(trajectory)
                yield trajectory, total_reward, total_steps
                # reset the trajectory
                trajectory = {k:[] for k in trajectory.keys()}

        # finally when we are actually done
        if len(trajectory['reward']) > 0:
            trajectory['state'].append(state) # append the terminal state
            trajectory = self._trajectory_to_tensor(trajectory)
            yield trajectory, total_reward, total_steps


    def _compute_Gt(self, rewards:Union[list, torch.Tensor]):
        ''' computes the partial returns at each time-step of the trajectory '''
        total_steps = len(rewards)
        partial_Gts = [0.0]*total_steps 
        prev_Gt = 0 # partial return from i+1 onwards
        for i in range(total_steps-1,-1,-1):
            ith_Gt = rewards[i] + self.gamma*prev_Gt # Gt from i onwards
            partial_Gts[i] = ith_Gt
            prev_Gt = ith_Gt
        partial_Gts = torch.tensor(partial_Gts, dtype=torch.float32, device=self.device)
        return partial_Gts


    def _take_steps(self, action:Tensor, render=False):
        """ This executes the action, and handels frame skipping.
        In frame skipping the same action is repeated and the observations
        and infos are are stored in a list. The next state (where the agent 
        lands) is computed using the make_state upon the list of observations
        and infos.
        
        If the episode terminates within a frame skip then the list is padded 
        using the last observed observation and info to maintain the same input
        size. 

        this function returns:
        nextState: the next state
        accumulatedReward: an accumulation of the rewards (currently sum)
        sumReward: the sum of the rewards seen 
        done:bool, whether the episode has ended 
        stepsTaken: the number of frames actually skipped - usefull when
                    the episode ends during a frame-skip """

        accumulatedReward = 0 # the reward the model will see
        sumReward = 0 # to keep track of the total reward in the episode
        stepsTaken = 0 # to keep track of the total steps in the episode

        observationList = []
        infoList = []

        for skipped_step in range(self.skipSteps):

            # repeate the action
            nextObservation, reward, done, info = self.env.step(action.item())
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


    def _value_model_grad_step(self, values:Tensor, partial_returns:Tensor):
        ''' compute loss and do single gradient step for value_model '''
        valueLoss = F.mse_loss(values.squeeze(), partial_returns.squeeze())
        self.value_optimizer.zero_grad()
        valueLoss.backward()
        clip_grads(self.value_model)
        self.value_optimizer.step()
        return valueLoss.item()


    def _trajectory_to_tensor(self, trajectory:'dict[str, list]') -> 'dict[str, Tensor]':
        ''' converts the trajectory from dict[str, list] to dict[str, Tensor]
        while mainitaining backprop path for log_prob and entropy entries '''
        for k in ['state', 'reward', 'action']:
            trajectory[k] = self._to_tensor(trajectory[k])
        trajectory['log_prob'] = torch.cat(tuple(trajectory['log_prob']))
        trajectory['entropy'] = torch.cat(tuple(trajectory['entropy']))
        return trajectory


    def _initBookKeeping(self):
        self.trainRewards = []
        self.episodeSteps = []
        self.policylosses = []
        self.valuelosses = []
        self.wallTime = []
        self.entropies = []


    def _performBookKeeping(self, trainReward, steps, policyloss, valueloss, timeElasped, entropy):
        self.trainRewards.append(trainReward)
        self.episodeSteps.append(steps)
        self.policylosses.append(policyloss)
        self.valuelosses.append(valueloss)
        self.wallTime.append(timeElasped)
        self.entropies.append(entropy)


    def _to_tensor(self, data:'list[Tensor]', requires_grad = False):
        return torch.tensor(data, dtype=torch.float32, device=self.device, requires_grad=requires_grad)


    def evaluate(self, evalExplortionStrategy:Strategy, EvalEpisodes=1, render=False):
        """ Evaluate the model for EvalEpisodes number of episodes """

        evalRewards = []
        evalSteps = []

        for evalEpisode in range(EvalEpisodes):

            done = False
            observation = self.env.reset()
            info = None # no initial info from gym.Env.reset

            # counters
            steps = 0
            totalReward = 0

            # user defines this func to make a state from a list of observations and infos
            state = self.make_state([observation for _ in range(self.skipSteps)], [info for _ in range(self.skipSteps)])

            # render
            if render: self.env.render()

            while not done:

                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

                # take action
                action = evalExplortionStrategy.select_action(state_tensor)

                # repeat the action skipStep number of times
                nextState, accumulatedReward, sumReward, done, stepsTaken = self._take_steps(action, render=render)

                steps += stepsTaken
                totalReward += sumReward

                if self.MaxStepsPerEpisode and steps >= self.MaxStepsPerEpisode: break

                # update state
                state = nextState

                # render
                if render: self.env.render()

            # decay exploration strategy params
            evalExplortionStrategy.decay()

            # append total episode reward
            evalRewards.append(totalReward)
            evalSteps.append(steps)

            # print
            print(f"evalEpisode: {evalEpisode} -> reward: {totalReward} steps: {steps}")
        
        evalinfo = {'rewards': evalRewards, 'steps':evalSteps}
        return evalinfo
