from examples.VPGexample2 import run_VPG_on_cartpole_V0
from examples.helper_funcs.helper_funcs import run_big_experiment

if __name__ == "__main__":
    results = run_big_experiment(run_VPG_on_cartpole_V0, 5)