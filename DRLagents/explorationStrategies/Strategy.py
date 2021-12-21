from typing import Union
from torch import Tensor, nn

# base class for exploration strategy
class Strategy:
    ''' The base class for strategies '''
    def __init__(self, model:nn.Module, outputs_LogProbs=False) -> None:
        """ outputs_LogProbs: whether the model outputs 
                                the Log of action-probablities 
            NOTE: If outputs_LogProbs=False then it will be assumed the model
                    maps state to Q-values. """
        # define required params
        pass


    def select_action(self, state:Tensor, 
                        logProb_n_entropy=False, grad=False) \
                        -> Union[Tensor, 'tuple[Tensor, Tensor, Tensor]']:
        """ This method should be implemented in every derived class.
        Selects an action and also returns optional outputs.

        state: preferably a a 1x(the dimension of state) shaped Tensor

        logProb_n_entropy: when True also return the log-probablity of the 
                            selected action and entropy of the output 
                            probablity dist.
        
        grad: whther gradients should pass, this is False if logProb_n_entropy
                is False. Otherwise if grad is True then torch.no_grad() is not
                used.

        outputs -   action - if logProb_n_entropy is False
                    action, log_prob, entropy - otherwise

        NOTE: if outputs_probs=False, then a log-softmax is applied to the 
                model's output to get a proxy of action probablities. 
                This may not be a very good proxy.
        """
        raise NotImplementedError()


    def decay(self):
        """ called by the traning function after the end of 
        each episode. use this to update any paramters that
        should be updated once every episode """
        # decay params if applicable
        pass