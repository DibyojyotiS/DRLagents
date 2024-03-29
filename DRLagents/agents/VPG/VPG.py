# Reinforce with baseline (VPG) is implemented here 
# This particular version also uses entropy in the policy loss

import os
from time import perf_counter
import time
from typing import Union

import gym
from numpy.core.fromnumeric import mean
import torch
import torch.nn.functional as F
from DRLagents.agents.common_helper_funcs.helper_funcs import clip_grads, compute_GAE
from .explorationStrategies import Strategy
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from .explorationStrategies.greedyStrategy import greedyAction
from DRLagents.utils.helper_funcs import printDict

class VPG:
    ''' ## Vanila Policy Gradient with GAE

    action-advantage is estimated using the GAE
    Entropy is also added to the policy loss after 
    being scaled by a factor of beta. 
    
    It is also possible to not use GAE and rather
    estimate action-advantages using the partial
    returns and value-network directly. (This corresponds
    to lamda = 1 in case of GAE)'''

    def __init__(self, 
                # training necessities
                env:gym.Env, policy_model: nn.Module, value_model:nn.Module,
                trainExplortionStrategy: Strategy,
                policy_optimizer: Optimizer, 
                value_optimizer: Optimizer,

                # optional training necessities
                make_state = lambda trajectory,action_taken: trajectory[-1][0],
                gamma = 0.99,
                lamda = 0.8,
                beta = 0.2,
                MaxTrainEpisodes = 500,
                MaxStepsPerEpisode = None,
                trajectory_seg_length = None,
                value_steps = 10,

                # miscellaneous
                skipSteps = 0,
                breakAtReward = float('inf'),
                printFreq = 50,
                use_gae = True,
                eval_episode = None,
                evalExplortionStrategy: Union[Strategy, None]=None,
                shared_policy_value_nets = False,
                log_dir = None,
                user_printFn = lambda : None,
                save_snapshots = True,
                device= torch.device('cpu'),
                print_args=False) -> None:
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

        make_state: function that takes a trajectory (list of [next-observation, info, reward, done]) & action_taken to make a state.
                    This function should handle info, action_taken, reward, done as Nones, since gym.env.reset doesnot return info.
                    Should handle trajectoru of variable lengths.

        gamma: the discount factor

        lamda: the GAE lamda to interpolate

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

        use_gae: wether to compute and use the GAE returns to estimate action-advantage. If False,
                the returns are used instead to estimate the action-advantage by subtracting state-value

        eval_episode: evaluate the agent after every eval_episode-th episode

        evalExplortionStrategy: strategy to be used for evalutation : default - greedy

        shared_policy_value_nets: whether the policy and value nets share some parameters, in this case the policy and value-loss
                                    are combined together as: policy-loss + c1 * value-loss
        
        log_dir: path to the directory to save logs and models every print episode (if save_snapshots=True)
                set log_dir to None to save nothing

        user_printFn: user provides this function to print user stuff (called every printFreq episode)

        save_snapshots: see log_dir

        # Implementation notes:\n
        NOTE: policy_model maps the states to action log-probablities
        NOTE: It is not recommended to share layers between value and policy model for this implementation of VPG.
                See VPGexample2 for a sketchy way to share layers.
        NOTE: It is assumed that optimizer is already setup with the network parameters and learning rate.
        NOTE: The make_state function gets an input of a list of observations and infos corresponding to the skipped and the current states.
        NOTE: The initial list of observations is the initial observation repeated skipSteps+1 number of times.
        NOTE: If episode terminates while skipping, the list is padded with the last observation and info
        NOTE: The models are not updated within the skipSteps steps.
        NOTE: value_model returns a single value for each state """

        if print_args: printDict(self.__class__.__name__, locals())

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
        self.lamda = lamda
        self.beta  = beta
        self.MaxTrainEpisodes = MaxTrainEpisodes
        self.MaxStepsPerEpisode = MaxStepsPerEpisode
        self.trajectory_seg_length = trajectory_seg_length
        self.value_steps = value_steps

        # miscellaneous
        self.skipSteps = skipSteps + 1
        self.breakAtReward = breakAtReward
        self.printFreq = printFreq
        self.use_gae = use_gae
        self.device = device
        self.eval_episode = eval_episode
        self.shared_nets = shared_policy_value_nets
        self.save_snapshots = save_snapshots
        self.user_printFn = user_printFn

        self.log_dir = log_dir
        if log_dir is not None:
            self.log_dir = os.path.join(self.log_dir, 'trainLogs')
            if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

        # required inits
        self._initBookKeeping()

        if not evalExplortionStrategy:
            self.evalExplortionStrategy = greedyAction(outputs_LogProbs=True)
            print("Using greedy strategy as evalExplortionStrategy.")
        else:
            self.evalExplortionStrategy = evalExplortionStrategy 


    def trainAgent(self, render=False):
        """ The main function to train the policy and value models """
        
        timeStart = perf_counter()

        for episode in range(self.MaxTrainEpisodes):

            # get the trajectory segments & optimize models
            for output_tupple in self._genetate_trajectory(render):
                trajectory, totalReward, total_steps = output_tupple
                policyloss, valueloss, avgEntropy = self._optimizeAgent(trajectory)

            # do train book-keeping
            self._performTrainBookKeeping(episode, totalReward, total_steps, policyloss, 
                                        valueloss, perf_counter()-timeStart, avgEntropy)

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
                print(f'episode: {episode} -> reward: {totalReward}, steps:{total_steps} time-elasped: {perf_counter()-timeStart:.2f}s')
                if eval_done:
                    print(f'eval-episode: {episode} -> reward: {evalReward}, steps: {evalSteps}, wall-time: {evalWallTime}')
                self._save_snapshot(episode)
                self.user_printFn() # call the user-printing function

            # early breaking
            if totalReward >= self.breakAtReward:
                print(f'stopping at episode {episode}')
                break

        # print the last episode if not already printed
        if episode % self.printFreq != 0:
            print(f'episode: {episode} -> reward: {totalReward}, steps:{total_steps} time-elasped: {perf_counter()-timeStart:.2f}s')
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

                state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)

                # take action
                action = evalExplortionStrategy.select_action(self.policy_model, state_tensor)

                # repeat the action skipStep number of times
                nextState, sumReward, done, stepsTaken = self._take_steps(action)

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


    def _optimizeAgent(self, trajectory:'dict[str, Tensor]'):
        """ to optimize the agent """
        # compute the partial return at every step
        partial_returns = self._compute_Gt(trajectory['reward'])

        # action_probablities
        action_logprobs = trajectory['log_prob']
        mean_entropy = trajectory['entropy'].mean()

        # state-values
        values = self.value_model(trajectory['state']).squeeze(dim=-1)

        # compute the action-advantages
        action_advantages = partial_returns - values[:-1] if not self.use_gae else \
            compute_GAE(values, trajectory['reward'], trajectory['done'], self.gamma, self.lamda)

        # compute policy-loss
        policyLoss = - (action_advantages.detach()*action_logprobs).mean() + self.beta*mean_entropy

        # grad-step policy model
        self.policy_optimizer.zero_grad()
        policyLoss.backward()
        clip_grads(self.policy_model)
        self.policy_optimizer.step()

        # grad-step value model
        valueLoss = F.mse_loss(values[:-1].squeeze(), partial_returns.squeeze())
        self.value_optimizer.zero_grad()
        valueLoss.backward()
        clip_grads(self.value_model)
        self.value_optimizer.step()

        for value_step in range(1, min(self.value_steps, values.shape[0])):
            values = self.value_model(trajectory['state'][:-1]).squeeze()
            valueLoss = F.mse_loss(values.squeeze(), partial_returns.squeeze())
            self.value_optimizer.zero_grad()
            valueLoss.backward()
            clip_grads(self.value_model)
            self.value_optimizer.step()

        return policyLoss.item(), valueLoss.item(), mean_entropy.item()


    def _genetate_trajectory(self, render=False):
        """ 
        This is a generator function to generate trajectories

        It runs an episode and stores the history as a dict with the 
        keys 'state', 'action', 'reward', 'log_prob', 'entropy'
        returns: trajectory_dict, total_reward, total_steps
        NOTE: the states also include the terminal state, making
                its length 1 more than actions, rewards and 
                partial returns """
        
        trajectory = {'state':[], 'action':[], 'reward':[], 
                    'log_prob':[], 'entropy':[], 'done':[]}

        # bookeeping counters
        total_reward = 0.0
        total_steps = 0
        total_appends = 0 # track length of original trajectory

        done = None
        observation = self.env.reset()
        info = None # no initial info from gym.Env.reset

        # render
        if render: self.env.render()

        # user defines this func to make a state from a list of observations and infos
        state = self.make_state([[observation,info,None,done]], None)
        
        while not done:
            # take the action and handle frame skipping
            state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
            action, log_prob, entropy = self.trainExplortionStrategy.select_action(self.policy_model, 
                                                    state_tensor, grad=True, logProb_n_entropy=True)
            nextState, sumReward, done, stepsTaken = self._take_steps(action)
            # append to trajectory
            trajectory['state'].append(state)
            trajectory['action'].append(action)
            trajectory['log_prob'].append(log_prob) # log-probabliy of taken action
            trajectory['entropy'].append(entropy) # entropy of action probablities
            trajectory['reward'].append(sumReward)
            trajectory['done'].append(done)
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


    def _take_steps(self, action:Tensor):
        """ This executes the action, and handels frame skipping.
        In frame skipping the same action is repeated and the observations
        and infos are are stored in a list. The next state (where the agent 
        lands) is computed using the make_state upon the list of observations
        and infos.
        
        If the episode terminates within a frame skip then the list is padded 
        using the last observed observation and info to maintain the same input
        size. 

        ### parameters
        1. action: Tensor
                - action to repeate while frame-skipping

        ### returns:
        1. nextState: 
                - the next state
        2. sumReward: float
                - the sum of the rewards seen 
        3. done:bool
                - whether the episode has ended 
        4. stepsTaken: int
                - the number of frames actually skipped - usefull when
                    the episode ends during a frame-skip """

        action_taken = action.item()

        sumReward = 0 # to keep track of the total reward in the episode
        stepsTaken = 0 # to keep track of the total steps in the episode

        skip_trajectory = []

        for skipped_step in range(self.skipSteps):

            # repeate the action
            nextObservation, reward, done, info = self.env.step(action.item())
            sumReward += reward
            stepsTaken += 1

            skip_trajectory.append([nextObservation, info, reward, done])

            if done: break
        
        # compute the next state 
        nextState = self.make_state(skip_trajectory, action_taken)

        return nextState, sumReward, done, stepsTaken


    def _trajectory_to_tensor(self, trajectory:'dict[str, list]') -> 'dict[str, Tensor]':
        ''' converts the trajectory from dict[str, list] to dict[str, Tensor]
        while mainitaining backprop path for log_prob and entropy entries.
        uses torch.cat under the hood to retain backprop graph '''
        for k in ['state', 'reward', 'action']:
            trajectory[k] = self._to_tensor(trajectory[k])
        trajectory['log_prob'] = torch.cat(tuple(trajectory['log_prob']))
        trajectory['entropy'] = torch.cat(tuple(trajectory['entropy']))
        return trajectory


    def _to_tensor(self, data:'list[Tensor]', requires_grad = False):
        return torch.tensor(data, dtype=torch.float32, device=self.device, requires_grad=requires_grad)


    def _initBookKeeping(self):
        self.trainBook = {
            'episode':[], 'reward': [], 'steps': [],
            'policy-loss': [], 'value-loss':[], 'wallTime': [],
            'entropy': []
        }
        self.evalBook = {
            'episode':[], 'reward': [], 'steps': [],
            'wallTime': []
        }

        if self.log_dir is not None:
            # open the logging csv files
            self.trainBookCsv = open(os.path.join(self.log_dir, 'trainBook.csv'), 'w', 1, encoding='utf-8')
            self.evalBookCsv = open(os.path.join(self.log_dir, 'evalBook.csv'), 'w', 1, encoding='utf-8')
            self.trainBookCsv.write('episode, reward, steps, policy-loss, value-loss, wallTime, entropy\n')
            self.evalBookCsv.write('episode, reward, steps, wallTime\n')


    def _performTrainBookKeeping(self, episode, reward, steps, policyLoss, valueLoss, wallTime, _entropy):
        self.trainBook['episode'].append(episode)
        self.trainBook['reward'].append(reward)
        self.trainBook['steps'].append(steps)
        self.trainBook['policy-loss'].append(policyLoss)
        self.trainBook['value-loss'].append(valueLoss)
        self.trainBook['wallTime'].append(wallTime)
        self.trainBook['entropy'].append(_entropy)
        if self.log_dir is not None:
            self.trainBookCsv.write(f'{episode}, {reward}, {steps}, {policyLoss}, {valueLoss}, {wallTime}, {_entropy}\n')


    def _performEvalBookKeeping(self, episode, reward, steps, wallTime):
        self.evalBook['episode'].append(episode)
        self.evalBook['reward'].append(reward)
        self.evalBook['steps'].append(steps)
        self.evalBook['wallTime'].append(wallTime)
        if self.log_dir is not None:
            self.evalBookCsv.write(f'{episode}, {reward}, {steps}, {wallTime}\n')

    
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
        torch.save(self.policy_model.state_dict(), 
                    f'{self.log_dir}/policy_model_weights_episode_{episode}.pth')
        torch.save(self.value_model.state_dict(), 
                    f'{self.log_dir}/value_model_weights_episode_{episode}.pth')