from typing import Union
from torch import Tensor, nn

# base class for exploration strategy
class Strategy:
    ''' The base class for strategies '''
    def __init__(self, *args, **kwargs) -> None:
        """ """
        # define required params
        pass

    def select_action(self, qvalues:Tensor, *args, **kwargs) -> Tensor:
        """ This method should be implemented in every derived class.
        Selects an action and also returns optional outputs.

        qvalues: the q-value estimate from the model being trainied

        outputs: action
        """
        raise NotImplementedError()

    def decay(self):
        """ called by the traning function after the end of 
        each episode. use this to update any paramters that
        should be updated once every episode """
        # decay params if applicable
        pass

    def state_dict(self):
        """ all the non-callable params in __dict__ """
        return {k:v for k,v in self.__dict__.items() if not callable(v)}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)