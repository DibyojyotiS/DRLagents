# being implemented
from DRLagents.utils.helper_funcs import set_seed
from examples.D3QNexample import run_D3QN_on_cartpole_V0
from examples.VPGexample import run_VPG_on_cartpole_V0
from examples.helper_funcs.helper_funcs import run_big_experiment


if __name__ == "__main__":
    VPGresults = run_big_experiment(run_VPG_on_cartpole_V0, 5, 
                                    savedir='.temp_stuffs/VGP_stuff',
                                    show_plots=True)

    # PERD3QNresults = run_big_experiment(run_D3QN_on_cartpole_V0, 5, 
    #                                 savedir='.temp_stuffs/PERD3QN_stuff',
    #                                 show_plots=True)