from copy import deepcopy
from time import perf_counter

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
from ..common_helper_funcs import *

# things to discuss about skipping frames
# 1. how to accumulate the rewards in the skipped frames? -> currently a cumulative (summed) reward

class DQN:
    """ This class implements the DQN training algorithm. \n
    For discrete action space.
    One can use different optimizers, loss functions and replay buffers.
    Also used as a Base class for DDQN """

    def __init__(self, 
                # training necessities
                trainingEnv:gym.Env, model:nn.Module,
                trainExplorationStrategy: Strategy,
                optimizer: Optimizer,
                replayBuffer: ReplayBuffer,
                batchSize:int= 512,
                gamma:float= 0.8,
                update_freq:int= 5,
                MaxTrainEpisodes:int= 500,
                MaxStepsPerEpisode:int= None,

                # optional training necessities
                skipSteps = 0,
                optimize_every_kth_action = 1,
                num_gradient_steps = 1,
                make_state = default_make_state,
                make_transitions = default_make_transitions,
                loss = weighted_MSEloss,
                polyak_average = False,
                polyak_tau = 0.1,

                # eval while training
                evalFreq = None,
                evalExplortionStrategy= greedyAction(),

                # miscellaneous
                printFreq = 1,
                log_dir = None,
                snapshot_episode = 1,
                resumeable_snapshot:int = None,
                breakAtReward = float('inf'),
                device= torch.device('cpu'),
                float_dtype = torch.float32,
                print_args=False) -> None:
        '''
        Inits the training procedure. Also please read the notes at the bottom.

        ### training necessities
        
        1. trainingEnv: gym.Env
                - a gym env used for training on, that behaves similar to standard gym environments. 

        2. model: nn.Module
                - The deep network to be trained. It maps states to the action-values.

        3. trainExplortionStrategy: Strategy
                - training strategy similar to the classes defined in explorationStrategies.
                - see: DRLagents.explorationStrategies
                - it is possible to sub-class the base class Strategy 
                    and create your custom strategy.

        4. optimizer: torch.optim.Optimizer
                - any optimizer with same behaviour one from torch.optim

        5. replayBuffer: ReplayBuffer
                - a instance of the class ReplayBuffer (see DRLagents.replaybuffers)
                  like ExperienceReplayBuffer or PrioritizedReplayBuffer

        6. batchSize: int
                - the size of the batch of expericence tuples sampled from the replayBuffer

        7. gamma: float (default 5) 
                - the discount factor
                - used in computing the 1-step TD error

        8. update_freq: int (default 1)
                - if not None, the target model is updated every update_freq-th episode

        9. MaxTrainEpisodes: int (default 500)
                - maximum number of episodes to train

        10. MaxStepsPerEpisode: int (default None)
                - break the episode if number of steps taken reaches or exceeds MaxStepsPerEpisode.
                - if None then the episode is continued until trianingEnv.step(...) returns done as True.

        ### optional training necessities

        11. skipSteps: int (default 0)
                - the number of steps to repeate the previous action 
                - the model not optimized during skipped steps
                - usually the a state-transition thus produced as: 
                    [prev-state, action, sumed rewards, next-state, done]
                - you can provide a custom way of making transitions in **make_transitions**
                  argument explained later.

        12. optimize_every_kth_action: int (default 1)
                - the online model is update after every kth new action. 
                - To train the online model only at the end of an episode set this to -1

        13. num_gradient_steps: int (default 1)
                - the number of gradient updates every optimize_kth_step.
        
        14. make_state: function (default default_make_state)
                - inputs:
                    - trajectory: list of [next-observation, info, reward, done].
                        If skipSteps is 0, then trajectory will be of length 1.
                    - action: the action repeated during the trajectory. 
                - Since gym.Env.reset returns only the first observation, this function 
                    must handle [observation, None, None, None]. Though this only happens 
                    at the beginning of an episode after the call to trainEnv.reset
                    (or evalEnv.reset in evaluation).
                - This is used to draw the first most state.
                - Should also handle trajectory of variable lengths.

        15. make_transitions: function (default default_make_state)
                - inputs: (trajectory, state, action, mextState)
                        - trajectory: which is a list of [next-observation, info, reward, done].
                            If skipSteps is 0, then trajectory will be of length 1.
                        - state: the state before the begenning of the frame-skipping
                        - action: the action repeated for frame-skipping
                        - nextState: the state after the frame-skipping
                - creates a list of state-transitions of the form [state, action, reward, next-state, done]
                - Inputs are- trajectory, state, action, nextState
                                trajectory: which is a list of [observation, info, reward, done]
                                state: the state before the begenning of the frame-skipping
                                action: the action used during the frame-skipping
                                nextState: the state after the frame-skipping
                            Should handle trajectories of variable lengths.

        16. loss: function (default weighted_MSE)
                - inputs: (X, target, weights[optional])
                - if the replayBuffer.sample(...) outputs the sampleWeights 
                  The loss will be called as loss(Qestimate, td_target, weights=sampleWeights) 
                - otherwise loss will be called as loss(Qestimate, td_target).
                - For weighted_loss examples look in DRLagents.utils.weightedLosses

        17. polyak_average: bool (default False)
                - whether to do the polyak-avaraging of target model as 
                    polyak_tau*online_model + (1-polyak_tau)*target_model
                - if enabled, polyak-averaging will be done at end of every episode
        
        18. polyak_tau: float (default 0.1)
                - the target model is updated according to tau*online_model + (1-tau)*target_model 
                    in every episode

        ### eval while training

        19. evalFreq: int (default None)
                - if not None, evaluate the agent at every evalFreq-th episode

        20. evalExplortionStrategy: Strategy (default greedy-strategy)
                - strategy to select actions from q-values during evaluation

        ## miscellaneous 

        21. printFreq: int (default 1) 
                - print training progress every printFreq episode
        
        22. log_dir: str (default None)
                - path to the directory to save logs and models 
                - set log_dir to None to save nothing

        23. snapshot_episode: int (default 1) 
                - save intermidiate models in the log_dir every snapshot_episode-th episode. 
                - Set this to 0 or None to not snapshot. 
                - Or set to 1 (default) to snapshot every episode. 

        24. resumeable_snapshot: int (default None)
                - saves the (or overwrites the saved) replay-buffer, trainExporationStrategy, 
                    optimizer-state, for every resumeable_snapshot-th snapshot. 
                - set this to 0 or None to not save resumables
                - set to 1 to save resumables at every snapshot

        25. breakAtReward: float (default inf) 
                - break training when the reward reaches/crosses breakAtReward

        26. device: torch.device (default torch.device('cpu'))
                - the device the model is on

        27. float_dtype: 
                - the torch float dtype to be used

        28. print_args: bool (default False)
                - prints the names and values of the arguments (usefull for logging)

        ## Implementation notes:\n
            - It is assumed that optimizer is already setup with the network 
                parameters and learning rate.
            - assumes the keys as ['state', 'action', 'reward', 'nextState', 'done'] 
                in the sample dict from replayBuffer.
            - The make_state function gets an input of a list of observations and 
                infos corresponding to the skipped and the current states.
            - The model and replay buffer are not updated within the skipSteps steps.

        ## other comments: \n
        loss: Look in DRLagents.utils for examples of weighted losses. In case your 
        buffer would ONLY output samples, feel free to use losses like torch.nn.MSELoss. 
        Though all torch.nn losses may not be supported.
        '''

        if print_args: printDict(self.__class__.__name__, locals())

        # basic
        self.online_model = model
        self.target_model = deepcopy(model)
        self.trainingEnv = trainingEnv
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

        # eval
        self.evalFreq = evalFreq
        self.evalExplortionStrategy = evalExplortionStrategy
        
        # miscellaneous args
        self.skipSteps = skipSteps + 1
        self.polyak_average = polyak_average
        self.tau = polyak_tau
        self.breakAtReward = breakAtReward
        self.device = device
        self.printFreq = printFreq
        self.snapshot_episode = snapshot_episode
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
        self.current_episode = 0 # required if resuming the training


    def trainAgent(self, num_episodes:int=None, render=False, evalEnv=None,
                    train_printFn=None, eval_printFn=None):
        """
        The main function to train the model. If called more than once, 
        then it continues from the last episode. This functionality is usefull
        when we would like to do evaluation on a different environment.

        ### parameters
        1. num_episodes: int (default None)
                - the number of episodes to train for. 
                - If 0 or None, then this trains until 
                    MaxTrainEpisodes (passed in init)
                        
        2. render: bool (default False)
                - render the env using env.render() while training

        3. evalEnv: gym.Env (default None)
                - a gym.Env instance to evaluate on. 
                - If None, then the env passed in
                    init is used to perform the evaluation.

        4. train_printFn: function (default None)
                - user provides this function to print more stuff 
                - takes no arguments
                - called every printFreq episode

        5. eval_printFn: function (default None) 
                - user provides this function to print more stuff 
                - takes no arguments
                - called every evalFreq episode
        
        ### returns
            - train_history: dict[str, dict[str, list]]
                    - keys are 'train' and 'eval'
                    - train_history['train'] is a dict and contians record of
                    the reward, loss, steps and walltime for all train episodes
                    - train_history['eval'] is a dict and contains record of the
                    reward, steps and walltime for all the evaluation episodes
        """
        
        train_start_time = perf_counter()
        evalEnv = self.trainingEnv if evalEnv is None else evalEnv
        stop_episode = self.current_episode+num_episodes if num_episodes \
                        else self.MaxTrainEpisodes

        for episode in range(self.current_episode, stop_episode):
            
            done = False
            observation = self.trainingEnv.reset()
            info = None # no initial info from gym.Env.reset
            current_start_time = perf_counter()
            self.current_episode = episode
    
            # render
            if render: self.trainingEnv.render()

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
                nextState, skip_trajectory, sumReward, done, stepsTaken = self._take_steps(self.trainingEnv,action)

                # make transitions and push observation in replay buffer
                transitions = self.make_transitions(skip_trajectory, state, action, nextState)
                for _state, _action, _reward, _nextState, _done in transitions:
                    _state, _action, _reward, _nextState, _done = \
                            self._astensor(_state, _action, _reward, _nextState, _done)
                    self.replayBuffer.store(_state, _action, _reward, _nextState, _done)

                # update state, counters and optimize model
                state = nextState
                steps += stepsTaken; totalReward += sumReward; k += 1
                if not self.optimize_at_end and k % self.optimize_every_kth_action == 0: totalLoss += self._optimizeModel()
                if render: self.trainingEnv.render() # render
                if self.MaxStepsPerEpisode and steps >= self.MaxStepsPerEpisode: break

            # if required optimize at the episode end and compute average loss
            if not self.optimize_at_end:
                assert self.optimize_every_kth_action < k, 'REDUCE optimize_every_kth_action!!!'
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
            self._performTrainBookKeeping(episode, totalReward, steps, 
                                average_loss, perf_counter()-train_start_time)
            self._printTrainProgress(episode, totalReward, steps, current_start_time, 
                                    train_start_time, train_printFn)
            self._save_checkpoint_and_resumeables(episode)
            self._printEvalPogress(episode, eval_printFn, evalEnv)

            # early breaking
            if totalReward >= self.breakAtReward:
                print(f'stopping at episode {episode}')
                break
        # just cuirious to know the total time spent...
        print("total time elasped:", perf_counter() - train_start_time,'s')  
        
        return self._returnBook()


    def evaluate(self, evalExplortionStrategy:Strategy=greedyAction(), 
                EvalEpisodes=1, render=False, verbose=True, evalEnv:gym.Env=None):
        """ Evaluate the model for EvalEpisodes number of episodes.

        ### parameters
        1. evalExplortionStrategy: Strategy (default greedyAction)
                - the exploration strategy to use for evaluation.
        
        2. EvalEpisodes: int (default 1)
                - the number of times to do evaluation

        3. render: bool (default False)
                - render using env.render()
        
        4. verbose: bool (default True) 
                - print the evaluation result

        5. evalEnv: gym.Env (default None) 
                - the environment to evaluate on - to be used if the intended 
                env for evaluation is not the training env. If None (default) 
                then the env passed in the init is used for evaluation.

        ### returns
            - evalinfo: dict[str, list]
                    - contains the total-rewards, steps and wall-times for 
                    each episode of EvalEpisodes """

        evalRewards = []
        evalSteps = []
        wallTimes = []

        # run evals
        if not evalEnv: evalEnv = self.trainingEnv
        for evalEpisode in range(EvalEpisodes):
            done = False
            timeStart = perf_counter()
            # counters
            steps = 0
            totalReward = 0
            # user defines make_state func to make a state from a list of observations and infos
            observation,info = evalEnv.reset(),None # no initial info from usual gym.Env.reset
            state = self.make_state([[observation,info,None,done]], None)
            # render
            if render: evalEnv.render()
            while not done:
                # take action
                action = self.evalExplortionStrategy.select_action(self.online_model, torch.tensor([state], 
                                                                dtype=self.float_dtype, device=self.device))
                # take action and repeat the action skipStep number of times
                nextState, _, sumReward, done, stepsTaken = self._take_steps(evalEnv, action)
                # update state and counters
                state = nextState
                steps += stepsTaken
                totalReward += sumReward
                # break episode is required
                if self.MaxStepsPerEpisode and steps >= self.MaxStepsPerEpisode: break
                # render
                if render: evalEnv.render()
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
            # uncomment to get an idea of the speed
            # print(f'gradient-step {ith_gradient_update}', end='\r')

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

    def _take_steps(self, env:gym.Env, action:Tensor):
        """ This selects an action using the strategy and state, and then
        executes the action, and handels frame skipping.In frame skipping 
        the same action is repeated and the observations and infos are 
        stored in a list. The next state (where the agent lands) is computed 
        using the make_state upon the trajectory which is as described below.
        
        env: the env to step through using env.step(action)
        action: the action to repeate (for skip_steps+1 number of times)
        ---------------
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
            nextObservation, reward, done, info = env.step(action_taken)
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

    def _closeBookKeeping(self):
        # close the log files
        if self.log_dir is not None:
            self.trainBookCsv.close()
            self.evalBookCsv.close()

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
        return {'train': self.trainBook,
                'eval': self.evalBook}

    def _printTrainProgress(self, episode, totalReward, steps, 
            current_start_time, train_start_time, train_printFn):
        # show progress output
        override = (episode == self.MaxTrainEpisodes-1) # print the last episode always
        if self.printFreq and ((episode % self.printFreq == 0) or override):
            cur_time = perf_counter()
            time_taken = (cur_time - current_start_time)/60
            time_elasped = (perf_counter()-train_start_time)/60
            print(f'episode: {episode}/{self.MaxTrainEpisodes} -> reward: {totalReward},', 
                    f'steps:{steps}, time-taken: {time_taken:.2f}min,',
                    f'time-elasped: {time_elasped:.2f}min')
            if train_printFn is not None: train_printFn() # call the user-printing function

    def _printEvalPogress(self, episode, eval_printFn, evalEnv):
        # evaluate the agent and do eval-book keepings
        override = (episode == self.MaxTrainEpisodes-1)
        if self.evalFreq and ((episode % self.evalFreq == 0) or override):
            eval_info = self.evaluate(self.evalExplortionStrategy, 
                            EvalEpisodes=1, verbose=False, evalEnv=evalEnv)
            evalStats = [mean(eval_info[x]) for x in ['rewards','steps','wallTimes']]
            self._performEvalBookKeeping(episode, *evalStats)
            print('\n==================================================')
            print(f'eval-episode: {episode} -> reward: {evalStats[0]},',
                f'steps: {evalStats[1]}, wall-time: {evalStats[2]:.2f}s')
            if eval_printFn is not None: eval_printFn()
            print('==================================================\n')

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

    def attempt_resume(self, resume_dir:str=None, 
                        reload_buffer=True, reload_optim=True, reload_tstrat=True):
        """ attempt to load the replayBuffer, optimizer, trainExplorationStrategy,
        episode-number, online-model and the target-model to resume the training. 

        ## parameters
        1. resume_dir: str (default None)
                - the log_dir of the run from which to resume. 
                - If None then the log_dir passed in init is considered.
        2. reload_buffer: bool
                - wether to restore the state-dict of the 
                replayBuffer passed in init
        3. reload_optim: bool
                - wether to restore the state-dict of the 
                optimizer passed in init
        4. reload_tstrat: bool
                - wether to restore the state-dict of the 
                trainExplorationStrategy passed in init
        """

        if not resume_dir: resume_dir = self.log_dir
        else: resume_dir = os.path.join(resume_dir, 'trainLogs')

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
        self.trainBook = load_csvbook(f'{resume_dir}/trainBook.csv')
        self.evalBook = load_csvbook(f'{resume_dir}/evalBook.csv')

        # load the replaybuffer, optimizer, trainStrategy
        path = f'{resume_dir}/resume'
        if reload_buffer:
            self.replayBuffer.load_state_dict(
                torch.load(f'{path}/replayBuffer_statedict.pt'))
        if reload_optim:
            self.optimizer.load_state_dict(
                torch.load(f'{path}/optimizer_statedict.pt'))
        if reload_tstrat:
            self.trainExplorationStrategy.load_state_dict(
                torch.load(f'{path}/trainExplorationStrategy_statedict.pt'))

        # load the models
        with open(f'{resume_dir}/resume/episode.txt') as f: episode = int(f.readline().strip())
        path = f'{resume_dir}/models/episode-{episode}'
        self.online_model.load_state_dict(torch.load(f'{path}/onlinemodel_statedict.pt'))
        self.target_model.load_state_dict(torch.load(f'{path}/targetmodel_statedict.pt'))

        # update the start-episode
        self.current_episode = episode + 1

        return print(f'Successfully loaded stuff from {resume_dir}!',
                    '\nReady to resume training.')


    def __delete__(self):
        # close all IO stuff
        self._closeBookKeeping()