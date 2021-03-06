# being implemented
from DRLagents.utils.helper_funcs import set_seed
from examples.helper_funcs.bigExperiment import run_big_experiment

from examples.D3QNexample import run_D3QN_on_cartpole_V0
from examples.DQNexample import run_DQN_on_cartpole_V0
# from examples.VPGexample import run_VPG_on_cartpole_V0
from examples.VPGexample2 import run_VPG_on_cartpole_V0

if __name__ == "__main__":
    VPGresults = run_big_experiment(run_VPG_on_cartpole_V0, 5, 
                                    savedir='.temp_stuffs/VGP_stuff',
                                    show_plots=True)

    # PERD3QNresults = run_big_experiment(run_D3QN_on_cartpole_V0, 5, 
    #                                 savedir='.temp_stuffs/PERD3QN_stuff',
    #                                 show_plots=True)

    # DQNresults = run_big_experiment(run_DQN_on_cartpole_V0, 5, 
    #                                 savedir='.temp_stuffs/DQN_stuff',
    #                                 show_plots=True)
