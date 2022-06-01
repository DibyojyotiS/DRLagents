import torch
from torch import Tensor, nn

def polyak_update(online_model:nn.Module, target_model:nn.Module, tau=0.1):
    """ sets all the parameters of the target_model as 
        tau * online_model + (1 - tau) * target_model """
    with torch.no_grad():
        online_stateDict = online_model.state_dict()
        target_stateDict = target_model.state_dict()
        for key in online_stateDict:
            target_stateDict[key] = (1 - tau)*target_stateDict[key] + tau*online_stateDict[key]
        target_model.load_state_dict(target_stateDict)

def clip_grads(model:nn.Module, _min=-1.0, _max=1.0):
    for param in model.parameters():
        if param.grad is None: continue
        param.grad.clamp(_min, _max)


# GAE estimate
def compute_GAE(values:Tensor, rewards:list, dones:list, gamma:float, lamda:float):
    '''
    This computes the Generalized Advantage Estimate
    ### parameters
    1. value_estimates: Tensor
            - values from value_model on the trajectory
    2. rewards: list[float]
            - the rewards encountered in the trajectory
    3. gamma: float
            - the discount factor
    4. lamda: float
            - used to interpolate between 1-step TD
            and n-step TD

    ### returns
    - gae: Generalized Advantage Estimate
    '''

    T = len(rewards)
    gae = torch.zeros_like(values)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma*(not dones[t])*values[t+1] - values[t]
        gae[t] = delta + gamma*lamda*gae[t+1]*(t!=T-1)

    return gae


def default_make_transitions(trajectory, state, action, nextState):
    '''
    Usefull for frame-skipping, where the SAME action is repeated for a small number of steps
    
    ### parameters
    1. trajectory: list
            - which is a list of [next-observation, info, reward, done]
            - this is the trajectory while skipping-frames
            - all the observtions, infos, etc are those of the skipped frames
    2. state: 
            - the state before the beginning of the frame-skipping
    3. action: 
            - the action repeated for frame-skipping
    4. nextState: 
            - the state after the frame-skipping
    
    ### returns:
        - transitions: list
                - state-transitions of form [state, action, reward, next-state, done]
    '''
    reward = sum([r for o,i,r,d in trajectory])
    done = trajectory[-1][-1]
    return [[state, action, reward, nextState, done]]


def default_make_state(trajectory:list, action_taken):
    '''
    Usefull for frame-skipping, where the SAME action is repeated for a small number of steps
    
    ### parameters
    1. trajectory: list
            - which is a list of [next-observation, info, reward, done]
            - this is the trajectory while skipping-frames
            - all the observtions, infos, etc are those of the skipped frames
    2. action_taken:
            - the action repeated for frame-skipping
    
    ### returns
        - next-state: tensor
    ''' 
    return trajectory[-1][0]