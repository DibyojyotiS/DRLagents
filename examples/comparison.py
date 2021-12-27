# being implemented
from DRLagents.utils.helper_funcs import set_seed
from examples.DQNexample import run_DQN_on_cartpole_V0
from examples.VPGexample2 import run_VPG_on_cartpole_V0
from examples.helper_funcs.helper_funcs import run_big_experiment

set_seed(0)

if __name__ == "__main__":
    VPGresults = run_big_experiment(run_VPG_on_cartpole_V0, 5, 
                                    savedir='.temp_stuffs/VGPstuff',
                                    show_plots=True)