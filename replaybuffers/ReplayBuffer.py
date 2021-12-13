# base class for replay buffers all replay buffers must have the functions store and sample

import torch


class ReplayBuffer:

    def __init__(self) -> None:
        pass

    def store(self, *args, **kwargs):
        raise NotImplementedError()


    def sample(self, batchSize, **kwargs):
        raise NotImplementedError()

    
    def update(self, *args, **kwargs):
        pass # to be implemented only if updating is required


    def update_params(self):
        """ called in the training loop after each episode to update 
        parameters that should be updated once per episode.
        Doesnot take any arguments and returns nothing """
        pass


    def _lazy_buffer_init(self, experience, tuppleDesc):
        """ inits the buffer as a dict with the keys as in _tuppleDesc and the values
        as torch.empty tensor of length bufferSize and correct dimensions. The given experience
        is a list of tensors. And is used to infer the dimentions and the devices of the tensors.
        NOTE: Each element of the list experiece must be torch tensors """
        return {x: torch.empty(size=(self.bufferSize, *experience[i].shape), 
                                        dtype=experience[i].dtype,
                                        requires_grad=False,
                                        device=experience[i].device) 
                        for i,x in enumerate(tuppleDesc)}