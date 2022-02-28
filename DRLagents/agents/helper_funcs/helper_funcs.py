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
        # for paramOnline, paramTarget in zip(online_model.parameters(), target_model.parameters()):
        #     paramTarget.data = tau * paramOnline.data + (1 - tau) * paramTarget.data


def clip_grads(model:nn.Module, _min=-1.0, _max=1.0):
    for param in model.parameters():
        if param.grad is None: continue
        param.grad.clamp(_min, _max)


# GAE estimate
def compute_GAE(values:Tensor, rewards, gamma:float, lamda:float):
    '''
    value_estimates: values from value_model on the trajectory
    rewards: the rewards encountered in the trajectory
    gamma: discount factor
    lamda: i danced on the moon
    Assumes that the last value belongs to a terminal state
    '''

    T = len(rewards) # last state is terminal

    gae = torch.zeros_like(values)
    
    for t in reversed(range(T)):
        delta = rewards[t] + gamma*(t!=T-1)*values[t+1] - values[t]
        gae[t] = delta + gamma*lamda*gae[t+1]*(t!=T-1)

    return gae


def make_transitions(trajectory, state, nextState):
    '''
    Usefull for frame-skipping, where the SAME action is repeated for a small number of steps
    
    trajectory: which is a list of [next-observation, info, action-taken, reward, done]
    state: the state before the begenning of the frame-skipping
    nextState: the state after the frame-skipping
    
    returns:
    transitions: list of state-transitions of form [state, action, reward, next-state, done]
    '''
    reward = sum([r for o,i,a,r,d in trajectory])
    done = trajectory[-1][4]
    action = trajectory[-1][2]
    return [[state, action, reward, nextState, done]]
