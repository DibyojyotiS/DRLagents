from copy import deepcopy
from time import perf_counter
from typing import Union

import os
import gym
from numpy.core.fromnumeric import mean
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from DRLagents.explorationStrategies import Strategy, greedyAction
from DRLagents.replaybuffers import ReplayBuffer
from DRLagents.utils.weightedLosses import weighted_MSEloss
from DRLagents.utils.helper_funcs import printDict
from .helper_funcs import *

# things to discuss about skipping frames
# 1. how to accumulate the rewards in the skipped frames? -> currently a cumulative (summed) reward

class DQN:
    """ This class implements the DQN training algorithm. \n
    For discrete action space.
    One can use different optimizers, loss functions and replay buffers.
    Also used as a Base class for DDQN """

    def __init__(self, 
                # training necessities
                env:gym.Env, model:nn.Module,
                trainExplorationStrategy: Strategy,
                optimizer: Optimizer,
                replayBuffer: ReplayBuffer,
                batchSize = 512,
                gamma = 0.8,
                update_freq = 5,
                optimize_every_kth_action = 1,
                num_gradient_steps = 1,
                MaxTrainEpisodes = 500,
                MaxStepsPerEpisode = None,

                # optional training necessities
                skipSteps = 0,
                make_state = default_make_state,
                make_transitions = default_make_transitions,
                loss = weighted_MSEloss,
                polyak_average = False,
                polyak_tau = 0.1,

                # eval while training
                eval_episode = None,
                evalExplortionStrategy: Union[Strategy, None]=None,

                # miscellaneous
                log_dir = None,
                snapshot_episode = 1,
                resumeable_snapshot = 1,
                printFreq = 50,
                user_printFn = None,
                breakAtReward = float('inf'),
                device= torch.device('cpu'),
                float_dtype = torch.float32,
                print_args=False) -> None:
        '''
        # Psss note the notes at the bottom too.


        ## training necessities
        
        model: The deep network to be trained. It maps states to the action-values.
        
        env: a gym env, that behaves similar to standard gym environments. 

        replayBuffer: a instance of the class ReplayBuffer 
                      (like ExperienceReplayBuffer or PrioritizedReplayBuffer)
        
        trainExplortionStrategy: training strategy similar to the classes defined in explorationStrategies.
        
        optimizer: any optimizer with same behaviour one from torch.optim
        
        gamma: the discount factor

        update_freq: if not None, the target model is updated every update_freq-th episode

        optimize_kth_step: the online model is update after every kth new action (i.e. steps excluding 
                            skip-step). To train the online model only at the end of an episode set 
                            this to -1

        num_gradient_steps: the number of gradient updates every optimize_kth_step.

        MaxTrainEpisodes: maximum number of episodes to train for

        MaxStepsPerEpisode: break the episode if number of steps taken reaches or exceeds this


        ## optional training necessities

        skipSteps: the number of steps to repeate the previous action (model not optimized at skipped steps)
        
        make_state: function that takes a trajectory (list of [next-observation, info, reward, done]) 
                    & action_taken to make a state. This function should handle info, action_taken, reward, 
                    done as Nones, since gym.env.reset doesnot return info. Should also handle trajectory of 
                    variable lengths.

        make_transitions: creates a list of state-transitions of the form 
                          [state, action, reward, next-state, done]
                            Inputs are- trajectory, state, action, nextState
                                trajectory: which is a list of [observation, info, reward, done]
                                state: the state before the begenning of the frame-skipping
                                action: the action used during the frame-skipping
                                nextState: the state after the frame-skipping
                            Should handle trajectories of variable lengths.

        loss: The loss will be called as loss(Qestimate, td_target, weights=sampleWeights) 
                if the replayBuffer outputs sampleWeights, otherwise as loss(Qestimate, td_target).
                For weighted_loss examples look in DRLagents.utils.weightedLosses

        polyak_average: whether to do the polyak-avaraging of target model as 
                        polyak_tau*online_model + (1-polyak_tau)*target_model
                        if enabled, polyak-averaging will be done in every episode
        
        polyak_tau: the target model is updated according to tau*online_model + (1-tau)*target_model 
                    in every episode, the target_model is also updated to the online_model's weights 
                    every update_freq episode.


        ## eval while training

        eval_episode: evaluate the agent at every eval_episode-th episode

        evalExplortionStrategy: strategy to be used for evalutation : default greedy-strategy


        ## miscellaneous 
        
        log_dir: path to the directory to save logs and models set log_dir to None to save nothing

        snapshot_episode: save intermidiate models in the log_dir every snapshot_episode-th 
                        episode. Set this to 0 to not snapshot. Or set to 1 (default) to 
                        snapshot every episode. 

        resumeable_snapshot: saves the (or overwrited the saved) replay-buffer, trainExporationStrategy, 
                        optimizer-state, for every resumeable_snapshot-th snapshot. for example if 
                        resumeable_snapshot=5, then the aformentioned things will be saved every at 5th 
                        snapshot saved. Default, resumable_snapshot=1
        
        printFreq: print episode data every printFreq episode

        user_printFn: user provides this function to print user stuff 
                        (called every printFreq episode), default: None
       
        breakAtReward: break training when the reward reaches this number

        device: the device the model is on (default: cpu)

        float_dtype: the torch float dtype to be used

        print_args: prints the names and values of the arguments (usefull for logging)

        # Implementation notes:\n
        NOTE: It is assumed that optimizer is already setup with the network parameters and learning rate.
        NOTE: assumes the keys as ['state', 'action', 'reward', 'nextState', 'done'] in the sample dict 
              from replayBuffer.
        NOTE: The make_state function gets an input of a list of observations and infos corresponding to
              the skipped and the current states.
        NOTE: The initial list of observations is the initial observation repeated skipSteps+1 number of times.
        NOTE: If episode terminates while skipping, the list is padded with the last observation and info
        NOTE: The model and replay buffer are not updated within the skipSteps steps.

        # other comments: \n
        loss: Look in DRLagents.utils for examples of weighted losses. In case your buffer would ONLY output samples,
        feel free to use losses like torch.nn.MSELoss. Though all torch.nn losses may not be supported.
        '''

        if print_args: printDict(self.__class__.__name__, locals())

        # basic
        self.online_model = model
        self.target_model = deepcopy(model)
        self.env = env
        self.trainExplorationStrategy = trainExplorationStrategy
        self.replayBuffer = replayBuffer
        self.lossfn = loss
        self.optimizer = optimizer

        # training necessities
        self.gamma = gamma
        self.make_state = make_state
        self.make_transitions = make_transitions
        self.MaxTrainEpisodes = MaxTrainEpisodes
        self.MaxStepsPerEpisode = MaxStepsPerEpisode
        self.batchSize = batchSize
        self.update_freq_episode = update_freq
        self.optimize_every_kth_action = optimize_every_kth_action
        self.num_gradient_steps = num_gradient_steps
        
        # miscellaneous args
        self.skipSteps = skipSteps + 1
        self.polyak_average = polyak_average
        self.tau = polyak_tau
        self.breakAtReward = breakAtReward
        self.device = device
        self.printFreq = printFreq
        self.eval_episode = eval_episode
        self.snapshot_episode = snapshot_episode
        self.user_printFn = user_printFn
        self.float_dtype = float_dtype
        self.resumeable_snapshot = resumeable_snapshot

        self.log_dir = log_dir
        if log_dir is not None:
            self.log_dir = os.path.join(self.log_dir, 'trainLogs')
            if not os.path.exists(self.log_dir): os.makedirs(f'{self.log_dir}/models')
            if resumeable_snapshot and not os.path.exists(f'{self.log_dir}/resume'): 
                os.makedirs(f'{self.log_dir}/resume')

        # required inits
        self.target_model.eval()
        self._initBookKeeping()
        self.optimize_at_end = optimize_every_kth_action==-1
        self.start_episode = 0 # required if resuming the training

        if not evalExplortionStrategy:
            self.evalExplortionStrategy = greedyAction()
            print("Using greedy strategy as evalExplortionStrategy.")
        else: self.evalExplortionStrategy = evalExplortionStrategy


    def trainAgent(self, render=False):
        """The main function to train the model"""
        
        train_start_time = perf_counter()

        for episode in range(self.start_episode, self.MaxTrainEpisodes):
            
            done = False
            observation = self.env.reset()
            info = None # no initial info from gym.Env.reset
            current_start_time = perf_counter()
    
            # render
            if render: self.env.render()

            # counters
            k = 0 # count the number of calls to select-action
            steps = 0
            totalReward = 0
            totalLoss = 0

            state = self.make_state([[observation,info,None,done]], None)

            while not done:
                # take action
                action = self.trainExplorationStrategy.select_action(self.online_model, 
                            torch.tensor([state], dtype=self.float_dtype, device=self.device))

                # select action and repeat the action skipStep number of times
                nextState, skip_trajectory, sumReward, done, stepsTaken = self._take_steps(action)

                # make transitions and push observation in replay buffer
                transitions = self.make_transitions(skip_trajectory, state, action, nextState)
                for _state, _action, _reward, _nextState, _done in transitions:
                    _state, _action, _reward, _nextState, _done = self._astensor(_state, _action, _reward, _nextState, _done)
                    self.replayBuffer.store(_state, _action, _reward, _nextState, _done)

                # update state, counters and optimize model
                state = nextState
                steps += stepsTaken; totalReward += sumReward; k += 1
                if not self.optimize_at_end and k % self.optimize_every_kth_action == 0: totalLoss += self._optimizeModel()
                if render: self.env.render() # render
                if self.MaxStepsPerEpisode and steps >= self.MaxStepsPerEpisode: break

            # if required optimize at the episode end and compute average loss
            if not self.optimize_at_end:
                if k < self.optimize_every_kth_action: print('REDUCE optimize_every_kth_action!!!')
                average_loss = totalLoss / (k//self.optimize_every_kth_action)
            else: average_loss = self._optimizeModel()

            # update target model
            if self.update_freq_episode and episode % self.update_freq_episode == 0:
                self.target_model.load_state_dict(self.online_model.state_dict())
            elif self.polyak_average: polyak_update(self.online_model, self.target_model, self.tau)

            # decay exploration strategy and replay buffer params
            self.trainExplorationStrategy.decay()
            self.replayBuffer.update_params()

            # do train-book keeping, print progress-output, eval-output, etc... & save stuff
            self._performTrainBookKeeping(episode, totalReward, steps, average_loss, perf_counter()-train_start_time)
            self._print_stuff(episode, totalReward, steps, current_start_time, train_start_time)
            self._save_checkpoint_and_resumeables(episode)

            # early breaking
            if totalReward >= self.breakAtReward:
                print(f'stopping at episode {episode}')
                break
        # just cuirious to know the total time spent...
        print("total time elasped:", perf_counter() - train_start_time,'s')  
        
        return self._returnBook()


    def evaluate(self, evalExplortionStrategy:Strategy=greedyAction(), EvalEpisodes=1, 
                render=False, verbose=True):
        """ Evaluate the model for EvalEpisodes number of episodes """

        evalRewards = []
        evalSteps = []
        wallTimes = []
        # run evals
        for evalEpisode in range(EvalEpisodes):
            done = False
            timeStart = perf_counter()
            # counters
            steps = 0
            totalReward = 0
            # user defines make_state func to make a state from a list of observations and infos
            observation,info = self.env.reset(),None # no initial info from usual gym.Env.reset
            state = self.make_state([[observation,info,None,done]], None)
            # render
            if render: self.env.render()
            while not done:
                # take action
                action = self.evalExplortionStrategy.select_action(self.online_model, torch.tensor([state], 
                                                                dtype=self.float_dtype, device=self.device))
                # take action and repeat the action skipStep number of times
                nextState, _, sumReward, done, stepsTaken = self._take_steps(action)
                # update state and counters
                state = nextState
                steps += stepsTaken
                totalReward += sumReward
                # break episode is required
                if self.MaxStepsPerEpisode and steps >= self.MaxStepsPerEpisode: break
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

        returns the average-loss (as a float type - used only for plotting purposes)
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

        return total_loss/self.num_gradient_steps

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
                sampleWeights = torch.tensor(sampleWeights, dtype=self.float_dtype, device=self.device)
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
        # init the train and eval books
        self.trainBook = {'episode':[], 'reward': [], 'steps': [],'loss': [], 'wallTime': []}
        self.evalBook = {'episode':[], 'reward': [], 'steps': [],'wallTime': []}
        # open the logging csv files
        if self.log_dir is not None:
            self.trainBookCsv = open(os.path.join(self.log_dir, 'trainBook.csv'), 'a', 1, encoding='utf-8')
            self.evalBookCsv = open(os.path.join(self.log_dir, 'evalBook.csv'), 'a', 1, encoding='utf-8')
            if os.stat(os.path.join(self.log_dir, 'trainBook.csv')).st_size == 0:
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
        state = torch.tensor(state, dtype=self.float_dtype, device=self.device, requires_grad=False)
        nextState = torch.tensor(nextState, dtype=self.float_dtype, device=self.device, requires_grad=False)
        action = torch.tensor([action], dtype=torch.int64, device=self.device, requires_grad=False) # extra axis for indexing purpose
        reward = torch.tensor([reward], dtype=self.float_dtype, device=self.device, requires_grad=False)
        done = torch.tensor([done], dtype=torch.int, device=self.device, requires_grad=False)
        return state, action, reward, nextState, done

    def _returnBook(self):
        if self.log_dir is not None:
            # close the log files
            self.trainBookCsv.close()
            self.evalBookCsv.close()
        return {'train': self.trainBook,
                'eval': self.evalBook}

    def _print_stuff(self, episode, totalReward, steps, current_start_time, train_start_time):
        override = (episode == self.MaxTrainEpisodes-1) # print the last episode always
        # show progress output
        if self.printFreq and ((episode % self.printFreq == 0) or override):
            cur_time = perf_counter()
            time_taken = (cur_time - current_start_time)/60
            time_elasped = (perf_counter()-train_start_time)/60
            print(f'episode: {episode}/{self.MaxTrainEpisodes} -> reward: {totalReward},', 
                f'steps:{steps}, time-taken: {time_taken:.2f}min, time-elasped: {time_elasped:.2f}min')
            if self.user_printFn is not None: self.user_printFn() # call the user-printing function
        # evaluate the agent and do eval-book keepings
        if self.eval_episode and ((episode % self.eval_episode == 0) or override):
            eval_info = self.evaluate(self.evalExplortionStrategy, verbose=False)
            evalStats = [mean(eval_info[x]) for x in ['rewards','steps','wallTimes']]
            self._performEvalBookKeeping(episode, *evalStats)
            print(f'eval-episode: {episode} -> reward: {evalStats[0]}, steps: {evalStats[1]}, wall-time: {evalStats[2]}')

    def _save_checkpoint_and_resumeables(self, episode):
        if not self.log_dir: return
        if self.snapshot_episode and episode % self.snapshot_episode == 0:
            timebegin = perf_counter()
            # save the models
            path = f'{self.log_dir}/models/episode-{episode}'
            if not os.path.exists(path): os.makedirs(path)
            torch.save(self.online_model.state_dict(), f'{path}/onlinemodel_statedict.pt')
            torch.save(self.target_model.state_dict(), f'{path}/targetmodel_statedict.pt')            
            # save the optimizer, episode-number and replay-buffer
            if self.resumeable_snapshot and episode % (self.resumeable_snapshot*self.snapshot_episode) == 0:
                path = f'{self.log_dir}/resume'
                with open(f'{path}/episode.txt', 'w') as f: f.write(f'{episode}')
                torch.save(self.replayBuffer.state_dict(), f'{path}/replayBuffer_statedict.pt')
                torch.save(self.optimizer.state_dict(), f'{path}/optimizer_statedict.pt')
                torch.save(self.trainExplorationStrategy.state_dict(), 
                                        f'{path}/trainExplorationStrategy_statedict.pt')
            print(f'\tTime taken saving stuff: {perf_counter()-timebegin:.2f}s') 

    def attempt_resume(self):
        """ attempt to load the replayBuffer, optimizer, trainExplorationStrategy,
        episode-number, online-model and the target-model to resume the training. """

        def load_csvbook(path):
            with open(path, 'r') as f:
                keys = f.readline().strip().split(',')
                book = {k.strip():[] for k in keys}
                for line in f.readlines():
                    values = line.strip().split(',')
                    for key, value in zip(keys, values): 
                        book[key.strip()].append(
                            float(value) if '.' in value else int(value))
            return book

        # reconstruct the books
        self.trainBook = load_csvbook(f'{self.log_dir}/trainBook.csv')
        self.evalBook = load_csvbook(f'{self.log_dir}/evalBook.csv')

        # load the replaybuffer, optimizer, trainStrategy
        path = f'{self.log_dir}/resume'
        self.replayBuffer.load_state_dict(torch.load(f'{path}/replayBuffer_statedict.pt'))
        self.optimizer.load_state_dict(torch.load(f'{path}/optimizer_statedict.pt'))
        self.trainExplorationStrategy.load_state_dict(
                            torch.load(f'{path}/trainExplorationStrategy_statedict.pt'))

        # load the models
        with open(f'{self.log_dir}/resume/episode.txt') as f: episode = int(f.readline().strip())
        path = f'{self.log_dir}/models/episode-{episode}'
        self.online_model.load_state_dict(torch.load(f'{path}/onlinemodel_statedict.pt'))
        self.target_model.load_state_dict(torch.load(f'{path}/targetmodel_statedict.pt'))

        # update the start-episode
        self.start_episode = episode + 1

        return print('Successfully loaded stuff! Ready to resume training.')