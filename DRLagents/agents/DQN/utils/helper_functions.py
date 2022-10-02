import os
from time import perf_counter
import torch


def split_replay_buffer_sample(sample, float_dtype, torch_device):
    """ splits the given sample from replayBuffer.sample() into 
        states, actions, rewards, nextStates, dones, indices, sampleWeights """
    # handling for uniform and prioritized buffers
    if type(sample) == tuple and len(sample) == 3:
        ## prioritized experience replay type buffer
        batch, indices, sampleWeights = sample
        if type(sampleWeights) == torch.Tensor:
            sampleWeights = sampleWeights.to(torch_device)
        else:
            sampleWeights = torch.tensor(sampleWeights, dtype=float_dtype, device=torch_device)
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


def save_model_checkpoint(online_model, target_model, episode, log_dir):
    """saves the state-dicts of the models to {log_dir}/models/episode-{episode}"""
    if not log_dir: return
    # save the models
    path = f'{log_dir}/models/episode-{episode}'
    if not os.path.exists(path): os.makedirs(path)
    torch.save(online_model.state_dict(), f'{path}/onlinemodel_statedict.pt')
    torch.save(target_model.state_dict(), f'{path}/targetmodel_statedict.pt')     


def make_resumable_snapshot(log_dir, online_model, target_model, episode, 
            replayBuffer, optimizer, trainExplorationStrategy, lr_scheduler):
    """saves the state-dicts of the models to {log_dir}/models/episode-{episode}
    and saves the episode number and state-dicts of the replayBuffer, optimizer, 
    trainExplorationStrategy, lr_scheduler to '{log_dir}/resume'"""
    # save the optimizer, episode-number and replay-buffer
    if not log_dir: return
    timebegin = perf_counter()
    path = f'{log_dir}/resume'

    save_model_checkpoint(online_model, target_model, episode, log_dir)
    with open(f'{path}/episode.txt', 'w') as f: f.write(f'{episode}')
    torch.save(replayBuffer.state_dict(), f'{path}/replayBuffer_statedict.pt')
    torch.save(optimizer.state_dict(), f'{path}/optimizer_statedict.pt')
    torch.save(trainExplorationStrategy.state_dict(), f'{path}/trainExplorationStrategy_statedict.pt')
    if lr_scheduler is not None: torch.save(lr_scheduler.state_dict(), f'{path}/lr_scheduler_statedict.pt')
    
    print(f'\tResumable checkpoint time-taken: {perf_counter()-timebegin:.2f}s') 