import torch
from torch import nn

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