from collections import defaultdict
from time import perf_counter
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from DRLagents.utils.helper_funcs import movingAverage


def make_worker(id, runfn):
    print('started run:',id)
    return runfn()


def run_big_experiment(runfn, numRuns:int=1, numProcesses=8,
                    movingavgon:Union[list,None]=['trainRewards'],
                    plot=True):
    # importing torch multiprocessing as local
    from torch.multiprocessing import Pool, set_start_method
    start_time = perf_counter()

    print(f"Running {numRuns} experiments")
    with Pool(min(numProcesses,numRuns)) as p:
        worker = partial(make_worker, runfn=runfn)
        results = p.map(worker, range(numRuns))

    train_stats = {}
    eval_stats = defaultdict(list)
    for trainHist, evalInfo in results:
        for key in trainHist.keys():
            data = np.array(trainHist[key])
            if key not in train_stats:
                train_stats[key] = {
                        'min': data, 'max': data,
                        'E[x^2]': (data**2)/numRuns, 
                        'E[x]': data/numRuns
                    }
            else:
                current = train_stats[key]
                updated_entry = {
                    'min': np.where(current['min'] < data, current['min'], data),
                    'max': np.where(current['max'] > data, current['min'], data),
                    'E[x]': current['E[x]'] + data/numRuns,
                    'E[x^2]': current['E[x^2]'] + (data**2)/numRuns
                }
                train_stats[key] = updated_entry
        
        for key in evalInfo.keys():
            data = list(evalInfo[key])
            eval_stats[key].extend(data)
    

    # compute std
    for key in train_stats.keys():
        mean = train_stats[key]['E[x]']
        Ex2 = train_stats[key]['E[x^2]']
        del train_stats[key]['E[x^2]']
        std = np.sqrt(Ex2 - mean)
        train_stats[key]['std'] = std

    if movingavgon is not None:
        for key in movingavgon:
            for mkey in train_stats[key].keys():
                train_stats[key][mkey] = movingAverage(train_stats[key][mkey])   

    print(f"total-time taken {(perf_counter() - start_time)/60:.2f} minutes")

    # plot stuff
    if plot:
        for key in train_stats.keys():
            data = train_stats[key]
            mean = data['E[x]']
            std = data['std']
            plt.plot(mean)
            plt.fill_between(range(len(mean)), data['min'], data['max'], alpha=0.2)
            plt.fill_between(range(len(mean)), mean-3*std, mean+3*std, alpha=0.5)
            plt.xlabel('episode')
            plt.ylabel(key)
            plt.show()
        
        for key in eval_stats.keys():
            data = eval_stats[key]
            plt.boxplot(data)
            plt.xlabel('eval-run')
            plt.ylabel(key)
            plt.show()
    
    return train_stats, eval_stats
