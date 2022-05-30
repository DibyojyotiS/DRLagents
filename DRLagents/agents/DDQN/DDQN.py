import torch
from ..DQN.DQN import DQN


class DDQN(DQN):
    """ This class implements the Double-DQN training algorithm. \n
        For discrete action space.
        DDQN can also be used to implement D3QN (Duelling Double-DQN) 
        algorithm by passing a duelling network in the parameter model.\n
        Also one can use different optimizers, loss functions and replay buffers. """

    # DDQN is very very similar to DQN with the only difference in the compuatation of td_error
    # so only modifying the calculation of td_error so modifying the compute_loss
    def _compute_loss_n_updateBuffer(self, states, actions, rewards, nextStates, dones, indices, sampleWeights):

        # compute td-error
        argmax_a_Q = self.online_model(nextStates).detach().max(-1, keepdims=True)[1]
        max_a_Q = self.target_model(nextStates).detach().gather(-1, argmax_a_Q) # max estimated-Q values from target net
        current_Q = self.online_model(states).gather(-1, actions)
        td_target = rewards + self.gamma*max_a_Q*(1-dones)
        td_error = (td_target - current_Q).squeeze()

        # scale the error by sampleWeights
        if sampleWeights is not None:
            loss = self.lossfn(current_Q, td_target, weights=sampleWeights)
        else:
            loss = self.lossfn(current_Q, td_target)
        
        # update replay buffer
        if indices is not None:
            self.replayBuffer.update(indices, torch.abs(td_error).detach())

        return loss