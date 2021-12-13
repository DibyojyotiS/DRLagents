import numpy as np

# beta schedule examples
def make_exponential_beta_schedule(beta_init=0.1, beta_rate=0.007):
    """ example of beta_schedule for PrioritizedExperienceReplay buffer
    use: 
        >> myBetaSchedule = make_exponential_beta_schedule(0.1, 0.008)
        >> replay_buffer = PrioritizedExperienceReplay(1000, beta_schedule=myBetaSchedule) """
    def exponential_beta_schedule(episode):
        return 1 - (1-beta_init)*np.math.exp(-episode*beta_rate)  
    return exponential_beta_schedule