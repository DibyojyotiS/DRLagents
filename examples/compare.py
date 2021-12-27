# being implemented

from examples.D3QNexample import run_D3QN_on_cartpole_V0
from examples.VPGexample2 import run_VPG_on_cartpole_V0
from examples.helper_funcs.helper_funcs import run_n_experiments

VPG_plots = run_n_experiments(run_VPG_on_cartpole_V0, 5)
# D3QN_plots = run_n_experiments(lambda:run_D3QN_on_cartpole_V0(buffertype='uniform'), 5)
# PERD3QN_plots = run_n_experiments(run_D3QN_on_cartpole_V0, 5)