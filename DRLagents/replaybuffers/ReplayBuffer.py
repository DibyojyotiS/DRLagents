# base class for replay buffers all replay buffers must have the functions store and sample

import pickle
import torch


class ReplayBuffer:
    """ Base Class for replay buffers """

    def __init__(self) -> None:
        pass

    def store(self, *args, **kwargs):
        """ stores the experience in the buffer. """
        raise NotImplementedError()


    def sample(self, batchSize, **kwargs):
        """ samples a batchSize number of experiences and returns a dict as 

        sample = {  'state':tf.Tensor, 'action':tf.Tensor, 'reward':tf.Tensor, 
                    'nextState':tf.Tensor, 'done':tf.Tensor} 

        you can return something else but then must modify the training 
        algorithm. """
        raise NotImplementedError()

    
    def update(self, *args, **kwargs):
        """ to be implemented only if updating something at every gradient step
        is required """
        pass


    def update_params(self):
        """ called in the training loop after each episode to update 
        parameters that should be updated once per episode.
        Doesnot take any arguments and returns nothing """
        pass


    def _lazy_buffer_init(self, experience, tuppleDesc):
        """ inits the buffer as a dict with the keys as in _tuppleDesc and the values
        as torch.empty Tensors of length bufferSize and correct dimensions. The given experience
        is a list of tensors. And is used to infer the dimentions and the devices of the tensors.
        NOTE: only to be used if the stuff to store has pre-determined shapes
        NOTE: Each element of the list experiece must be torch tensors 
        NOTE: requires_grad=False is assumed """
        return {x: torch.empty(size=(self.bufferSize, *experience[i].shape), 
                                        dtype=experience[i].dtype,
                                        requires_grad=False,
                                        device=experience[i].device) 
                        for i,x in enumerate(tuppleDesc)}


    def save_to_disk(self, path):
        """ save the replay-buffer to disk """
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)