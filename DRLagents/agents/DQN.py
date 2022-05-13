from copy import deepcopy
from time import perf_counter
from typing import Union

import os
import gym
from numpy.core.fromnumeric import mean
import torch
from torch import nn
from torch import optim

from DRLagents.agents.helper_funcs.helper_funcs import clip_grads, make_transitions
from DRLagents.explorationStrategies.greedyStrategy import greedyAction
from .helper_funcs import polyak_update

from DRLagents.explorationStrategies import Strategy
from DRLagents.replaybuffers import ReplayBuffer
from DRLagents.utils.weightedLosses import weighted_MSEloss

# things to discuss about skipping frames
# 1. how to accumulate the rewards in the skipped frames? -> currently a cumulative (summed) reward

class DQN:
    """ This class implements the DQN training algorithm. \n
    For discrete action space.
    One can use different optimizers, loss functions and replay buffers."""

    def __init__(self, 
                # training necessities
                env:gym.Env, model:nn.Module,
                trainExplortionStrategy: Strategy,
                optimizer: optim.Optimizer,
                replayBuffer: ReplayBuffer,
                batchSize:int,

                # optional training necessities
                make_state = lambda trajectory,action_taken: trajectory[-1][0],
                make_transitions = make_transitions,
                gamma=0.8,
                MaxTrainEpisodes = 500,
                MaxStepsPerEpisode = None,
                loss = weighted_MSEloss,
                update_freq = 5,
                optimize_kth_step = 1,
                num_gradient_steps = 1,

                # miscellaneous
                skipSteps = 0,
                polyak_average = False,
                polyak_tau = 0.1,
                breakAtReward = float('inf'),
                printFreq = 50,
                eval_episode = None,
                evalExplortionStrategy: Union[Strategy, None]=None,
                log_dir = '.temp',
                user_printFn = lambda : None,
                save_snapshots = True,
                device= torch.device('cpu')) -> None:
        '''
        # Psss note the notes at the bottom too.


        ## training necessities
        
        model: The deep network to be trained. It maps states to the action-values.
        
        env: a gym env, that behaves similar to standard gym environments. Currently tested on env like cartpole.

        replayBuffer: a instance of the class ReplayBuffer (like ExperienceReplayBuffer or PrioritizedReplayBuffer)
        
        trainExplortionStrategy: training strategy similar to the classes defined in explorationStrategies.
        
        optimizer: any optimizer with same behaviour one from torch.optim


        ## optional training necessities
        
        make_state: function that takes a trajectory (list of [next-observation, info, reward, done]) & action_taken to make a state.
                    This function should handle info, action_taken, reward, done as Nones, since gym.env.reset doesnot return info.
                    Should handle trajectoru of variable lengths.

        make_transitions: creates a list of state-transitions of the form [state, action, reward, next-state, done]
                            Inputs are- trajectory, state, action, nextState
                                trajectory: which is a list of [observation, info, reward, done]
                                state: the state before the begenning of the frame-skipping
                                action: the action used during the frame-skipping
                                nextState: the state after the frame-skipping
                            Should handle trajectories of variable lengths.
        
        gamma: the discount factor
        
        MaxTrainEpisodes: maximum number of episodes to train for

        MaxStepsPerEpisode: break the episode if number of steps taken reaches or exceeds this

        loss: The loss will be called as loss(Qestimate, td_target, weights=sampleWeights) 
                if the replayBuffer outputs sampleWeights, otherwise as loss(Qestimate, td_target).
                For weighted_loss examples look in DRLagents.utils.weightedLosses
                
        update_freq: target model is updated every update_freq episodes

        optimize_kth_step: the online model is update after every kth step (action). 
                            To train the online model only at the end of an episode set this to -1

        num_gradient_steps: the number of gradient updates every optimize_kth_step.


        ## miscellaneous 

        skipSteps: the number of steps to repeate the previous action (model not optimized at skipped steps)
        
        polyak_average: whether to do the exponential varsion of polyak-avaraging until the update of target model
        
        polyak_tau: the target model is updated according to tau*online_model + (1-tau)*target_model in every episode, 
                    the target_model is also updated to the online_model's weights every update_freq episode.
        
        breakAtReward: break training when the reward reaches this number
        
        printFreq: print episode reward every printFreq episode

        eval_episode: evaluate the agent after every eval_episode-th episode

        evalExplortionStrategy: strategy to be used for evalutation : default greedy-strategy

        log_dir: path to the directory to save logs and models every print episode (if save_snapshots=True)
                set log_dir to None to save nothing

        user_printFn: user provides this function to print user stuff (called every printFreq episode)

        save_snapshots: see log_dir

        # Implementation notes:\n
        NOTE: It is assumed that optimizer is already setup with the network parameters and learning rate.
        NOTE: assumes the keys as ['state', 'action', 'reward', 'nextState', 'done'] in the sample dict from replayBuffer
        NOTE: The make_state function gets an input of a list of observations and infos corresponding to the skipped and the current states.
        NOTE: The initial list of observations is the initial observation repeated skipSteps+1 number of times.
        NOTE: If episode terminates while skipping, the list is padded with the last observation and info
        NOTE: The model and replay buffer are not updated within the skipSteps steps.

        # other comments: \n
        loss: Look in DRLagents.utils for examples of weighted losses. In case your buffer would ONLY output samples,
        feel free to use losses like torch.nn.MSELoss. Though all torch.nn losses may not be supported.
        '''

        # basic
        self.online_model = model
        self.env = env
        self.trainExplortionStrategy = trainExplortionStrategy
        self.batchSize = batchSize
        self.optimizer = optimizer

        # training necessities
        self.target_model = deepcopy(model)
        self.gamma = gamma
        self.make_state = make_state
        self.make_transitions = make_transitions
        self.MaxTrainEpisodes = MaxTrainEpisodes
        self.MaxStepsPerEpisode = MaxStepsPerEpisode
        self.lossfn = loss
        self.update_freq = update_freq
        self.replayBuffer = replayBuffer
        self.optimize_kth_step = optimize_kth_step
        self.num_gradient_steps = num_gradient_steps
        
        # miscellaneous args
        self.skipSteps = skipSteps + 1
        self.polyak_average = polyak_average
        self.tau = polyak_tau
        self.breakAtReward = breakAtReward
        self.device = device
        self.printFreq = printFreq
        self.eval_episode = eval_episode
        self.save_snapshots = save_snapshots
        self.user_printFn = user_printFn

        self.log_dir = log_dir
        if log_dir is not None:
            self.log_dir = os.path.join(self.log_dir, 'trainLogs')
            if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

        # required inits
        self.target_model.eval()
        self._initBookKeeping()
        self.optimize_at_end = optimize_kth_step==-1

        if not evalExplortionStrategy:
            self.evalExplortionStrategy = greedyAction(self.online_model)
            print("Using greedy strategy as evalExplortionStrategy.")
        else:
            self.evalExplortionStrategy = evalExplortionStrategy


    def trainAgent(self, render=False):
        """The main function to train the model"""
        
        timeStart = perf_counter()

        for episode in range(self.MaxTrainEpisodes):
            
            done = False
            observation = self.env.reset()
            info = None # no initial info from gym.Env.reset
    
            # render
            if render: self.env.render()

            # counters
            steps = 0
            totalReward = 0
            totalLoss = 0

            state = self.make_state([[observation,info,None,done]], None)

            while not done:

                # take action
                action = self.trainExplortionStrategy.select_action(torch.tensor(state, dtype=torch.float32, device=self.device))

                # select action and repeat the action skipStep number of times
                nextState, skip_trajectory, sumReward, done, stepsTaken = self._take_steps(action)

                # make transitions and push observation in replay buffer
                transitions = self.make_transitions(skip_trajectory, state, action, nextState)
                for _state, _action, _reward, _nextState, _done in transitions:
                    _state, _action, _reward, _nextState, _done = self._astensor(_state, _action, _reward, _nextState, _done)
                    self.replayBuffer.store(_state, _action, _reward, _nextState, _done)

                # optimize model
                if not self.optimize_at_end and steps % self.optimize_kth_step == 0:
                    loss = self._optimizeModel()
                    totalLoss += loss

                steps += stepsTaken
                totalReward += sumReward

                if self.MaxStepsPerEpisode and steps >= self.MaxStepsPerEpisode: break

                # update state
                state = nextState

                # render
                if render: self.env.render()

            # if required optimize at the episode end
            if self.optimize_at_end:
                totalLoss = self._optimizeModel()

            # update target model
            if episode % self.update_freq == 0:
                self.target_model.load_state_dict(self.online_model.state_dict())
            elif self.polyak_average:
                polyak_update(self.online_model, self.target_model, self.tau)

            # decay exploration strategy params
            self.trainExplortionStrategy.decay()

            # update the replay buffer params
            self.replayBuffer.update_params()

            # do train-book keeping
            self._performTrainBookKeeping(episode, totalReward, steps, totalLoss, perf_counter()-timeStart)

            # evaluate the agent and do eval-book keeping
            eval_done=False
            if self.eval_episode and (episode+1)%self.eval_episode == 0:
                eval_info = self.evaluate(self.evalExplortionStrategy, verbose=False)
                evalReward = mean(eval_info['rewards'])
                evalSteps = mean(eval_info['steps'])
                evalWallTime = mean(eval_info['wallTimes'])
                self._performEvalBookKeeping(episode, evalReward, evalSteps, evalWallTime)
                eval_done = True

            # show progress output
            if episode % self.printFreq == 0:
                print(f'episode: {episode} -> reward: {totalReward}, steps:{steps}, time-elasped: {perf_counter()-timeStart:.2f}s')
                if eval_done: print(f'eval-episode: {episode} -> reward: {evalReward}, steps: {evalSteps}, wall-time: {evalWallTime}')
                self._save_snapshot(episode)
                self.user_printFn() # call the user-printing function

            # early breaking
            if totalReward >= self.breakAtReward:
                print(f'stopping at episode {episode}')
                break

        # print the last episode if not already printed
        if episode % self.printFreq != 0:
            print(f'episode: {episode} -> reward: {totalReward}, steps:{steps} time-elasped: {perf_counter()-timeStart:.2f}s')
            if eval_done:
                print(f'eval-episode: {episode} -> reward: {evalReward}, steps: {evalSteps}, wall-time: {evalWallTime}')
            self._save_snapshot(episode)

        # just cuirious to know the total time spent...
        print("total time elasped:", perf_counter() - timeStart,'s')    
        
        return self._returnBook()


    def evaluate(self, evalExplortionStrategy:Strategy, EvalEpisodes=1, render=False, verbose=True):
        """ Evaluate the model for EvalEpisodes number of episodes """

        evalRewards = []
        evalSteps = []
        wallTimes = []

        for evalEpisode in range(EvalEpisodes):

            done = False
            timeStart = perf_counter()
            observation = self.env.reset()
            info = None # no initial info from gym.Env.reset

            # counters
            steps = 0
            totalReward = 0

            # user defines this func to make a state from a list of observations and infos
            state = self.make_state([[observation,info,None,done]], None)

            # render
            if render: self.env.render()

            while not done:

                # take action
                action = self.evalExplortionStrategy.select_action(torch.tensor(state, dtype=torch.float32, device=self.device))

                # take action and repeat the action skipStep number of times
                nextState, _, sumReward, done, stepsTaken = self._take_steps(action)

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
            wallTimes.append(perf_counter() - timeStart)

            # print
            if verbose:
                print(f"evalEpisode: {evalEpisode} -> reward: {totalReward} steps: {steps}")
        
        evalinfo = {'rewards': evalRewards, 'steps':evalSteps, 'wallTimes':wallTimes}
        return evalinfo


    def _compute_loss_n_updateBuffer(self, states, actions, rewards, nextStates, dones, indices, sampleWeights):
        """ this is the function used to compute the loss and is called within _optimizeModel. And this function
        also calls the self.replayBuffer.update(indices, priorities). Priorities can be something like 
        torch.abs(td_error).detach(). Or any other way you can think of ranking the samples. 
        
        Subclassing: Modify this function if only the loss computation has to be changed. 

        NOTE: that the indices, sampleWeights can be None if the replayBuffer doesnot return these.        
        NOTE: If indices is None then no need to update replayBuffer. Do something like:
            ...
            if indices is not None:
                self.replayBuffer.update(indices, torch.abs(td_error).detach())
        """

        # compute td-error
        max_a_Q = self.target_model(nextStates).detach().max(-1, keepdims=True)[0] # max estimated-Q values from target net
        current_Q = self.online_model(states).gather(-1, actions)
        td_target = rewards + self.gamma * max_a_Q*(1-dones)

        # scale the error by sampleWeights
        if sampleWeights is not None:
            loss = self.lossfn(current_Q, td_target, weights=sampleWeights)
        else:
            loss = self.lossfn(current_Q, td_target)
        
        # update replay buffer
        if indices is not None:
            td_error = (td_target - current_Q).squeeze()
            self.replayBuffer.update(indices, torch.abs(td_error).detach())

        return loss


    def _optimizeModel(self):
        """ 
        For those deriving something from DQN class: This function only implements the
        one step backprop, however the loss is computed seperatedly in the _compute_loss function
        which returns a scalar tensor representing the loss. If only the computation of loss has to 
        be modified then no need to modify _optimizeModel, modify the _compute_loss function 

        returns the loss (as a float type - used only for plotting purposes)
        """
        
        if len(self.replayBuffer) < self.batchSize: return 0

        total_loss = 0

        # do self.num_gradient_steps gradient updates
        for ith_gradient_update in range(self.num_gradient_steps):

            # sample a batch from ReplayBuffer
            sample = self.replayBuffer.sample(self.batchSize)
            states, actions, rewards, nextStates, dones, indices, sampleWeights = self._split_sample(sample)

            # compute the loss
            loss = self._compute_loss_n_updateBuffer(states, actions, rewards, nextStates, dones, indices, sampleWeights)

            # minimize loss
            self.optimizer.zero_grad()
            loss.backward()
            clip_grads(self.online_model)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss


    def _split_sample(self, sample):
        """ splits the given sample from replayBuffer.sample() into 
            states, actions, rewards, nextStates, dones, indices, sampleWeights """

        # handling for uniform and prioritized buffers
        if type(sample) == tuple and len(sample) == 3:
            ## prioritized experience replay type buffer
            batch, indices, sampleWeights = sample
            if type(sampleWeights) == torch.Tensor:
                sampleWeights = sampleWeights.to(self.device)
            else:
                sampleWeights = torch.tensor(sampleWeights, dtype=torch.float32, device=self.device)
        elif type(sample) == dict:
            ## expericence replay buffer with uniform sampling
            batch = sample
            indices = sampleWeights = None
        else:
            raise AssertionError('replayBuffer.sample() was expected to \
                return a tupple of size-3 (batch (dict), indices, sampleWeights) or only the batch.')

        # splits the values
        states, actions = batch['state'], batch['action']
        rewards, nextStates, dones = batch['reward'], batch['nextState'], batch['done']

        return states, actions, rewards, nextStates, dones, indices, sampleWeights


    def _take_steps(self, action):
        """ This selects an action using the strategy and state, and then
        executes the action, and handels frame skipping.In frame skipping 
        the same action is repeated and the observations and infos are 
        stored in a list. The next state (where the agent lands) is computed 
        using the make_state upon the trajectory which is as described below.
        
        this function returns:
        nextState: the next state
        trajectory: a list of [next-observation, info, action-taken, reward, done]
        sumReward: the sum of the rewards seen 
        done:bool, whether the episode has ended 
        stepsTaken: the number of frames actually skipped - usefull when
                    the episode ends during a frame-skip """
        
        action_taken = action.item()

        sumReward = 0 # to keep track of the total reward in the episode
        stepsTaken = 0 # to keep track of the total steps in the episode

        skip_trajectory = []

        for skipped_step in range(self.skipSteps):

            # repeate the action
            nextObservation, reward, done, info = self.env.step(action_taken)
            sumReward += reward
            stepsTaken += 1

            skip_trajectory.append([nextObservation, info, reward, done])

            if done: break
        
        # compute the next state 
        nextState = self.make_state(skip_trajectory, action_taken)

        return nextState, skip_trajectory, sumReward, done, stepsTaken


    def _initBookKeeping(self):
        self.trainBook = {
            'episode':[], 'reward': [], 'steps': [],
            'loss': [], 'wallTime': []
        }
        self.evalBook = {
            'episode':[], 'reward': [], 'steps': [],
            'wallTime': []
        }

        if self.log_dir is not None:
            # open the logging csv files
            self.trainBookCsv = open(os.path.join(self.log_dir, 'trainBook.csv'), 'w', 1, encoding='utf-8')
            self.evalBookCsv = open(os.path.join(self.log_dir, 'evalBook.csv'), 'w', 1, encoding='utf-8')
            self.trainBookCsv.write('episode, reward, steps, loss, wallTime\n')
            self.evalBookCsv.write('episode, reward, steps, wallTime\n')



    def _performTrainBookKeeping(self, episode, reward, steps, loss, wallTime):
        self.trainBook['episode'].append(episode)
        self.trainBook['reward'].append(reward)
        self.trainBook['steps'].append(steps)
        self.trainBook['loss'].append(loss)
        self.trainBook['wallTime'].append(wallTime)
        if self.log_dir is not None:
            self.trainBookCsv.write(f'{episode}, {reward}, {steps}, {loss}, {wallTime}\n')



    def _performEvalBookKeeping(self, episode, reward, steps, wallTime):
        self.evalBook['episode'].append(episode)
        self.evalBook['reward'].append(reward)
        self.evalBook['steps'].append(steps)
        self.evalBook['wallTime'].append(wallTime)
        if self.log_dir is not None:
            self.evalBookCsv.write(f'{episode}, {reward}, {steps}, {wallTime}\n')


    def _astensor(self, state, action, reward, nextState, done):
        state = torch.tensor(state, dtype=torch.float32, device=self.device, requires_grad=False)
        nextState = torch.tensor(nextState, dtype=torch.float32, device=self.device, requires_grad=False)
        action = torch.tensor([action], dtype=torch.int64, device=self.device, requires_grad=False) # extra axis for indexing purpose
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device, requires_grad=False)
        done = torch.tensor([done], dtype=torch.int, device=self.device, requires_grad=False)
        return state, action, reward, nextState, done

    
    def _returnBook(self):
        if self.log_dir is not None:
            # close the log files
            self.trainBookCsv.close()
            self.evalBookCsv.close()
        return {
            'train': self.trainBook,
            'eval': self.evalBook
        }


    def _save_snapshot(self, episode):
        if not self.log_dir or not self.save_snapshots: return
        torch.save(self.online_model.state_dict(), 
                    f'{self.log_dir}/onlinemodel_weights_episode_{episode}.pth')
        torch.save(self.target_model.state_dict(), 
                    f'{self.log_dir}/targetmodel_weights_episode_{episode}.pth')