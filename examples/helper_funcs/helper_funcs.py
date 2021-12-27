from collections import defaultdict
from time import perf_counter
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import torch

from DRLagents.utils.helper_funcs import movingAverage


def run_n_experiments(runfn, numRuns:int=1, 
                    movingavgon:Union[list,None]=['trainRewards'],
                    plot=True):

    print(f"Running {numRuns} experiments")

    def first_entry(data):
        x = {'min': data, 'mean': data/numRuns, 'max': data}
        return x

    def new_entry(current_data, data):
        temp = current_data['min']
        current_data['min'] = np.where(temp < data, temp, data)
        temp = current_data['max']
        current_data['max'] = np.where(temp > data, temp, data)
        current_data['mean'] += data/numRuns
        return current_data

    # aggegrate results
    train_print_data = {}
    eval_print_data = {}
    start_time = perf_counter()
    for n in range(numRuns):
        print(f"Run-{n}")
        trainHist, evalInfo = runfn()

        for key in trainHist.keys():
            data = np.array(trainHist[key])
            if key not in train_print_data:
                train_print_data[key] = first_entry(data)
            else:
                train_print_data[key] = new_entry(train_print_data[key], data)
        
        for key in evalInfo.keys():
            data = list(evalInfo[key])
            if key not in eval_print_data:
                eval_print_data[key] = []
            eval_print_data[key].extend(data)
    
    # smooth 
    if movingavgon is not None:
        for key in movingavgon:
            for mkey in train_print_data[key].keys():
                train_print_data[key][mkey] = movingAverage(train_print_data[key][mkey])

    print(f"total-time taken {(perf_counter() - start_time)/60:.2f} minutes")

    if plot:
        for key in train_print_data.keys():
            data = train_print_data[key]
            plt.plot(data['mean'])
            plt.fill_between(x = range(len(data['mean'])), y2=data['min'], y1=data['max'], alpha=0.5)
            plt.xlabel('episode')
            plt.ylabel(key)
            plt.show()
        
        for key in eval_print_data.keys():
            data = eval_print_data[key]
            plt.boxplot(data)
            plt.xlabel('eval-run')
            plt.ylabel(key)
            plt.show()

    return train_print_data, eval_print_data



def run_big_experiment(runfn, numRuns:int=1, 
                    movingavgon:Union[list,None]=['trainRewards'],
                    plot=True):
    # importing torch multiprocessing as local
    from torch.multiprocessing import Pool, set_start_method
    set_start_method('spawn')
    start_time = perf_counter()

    def worker(x):
        trainHistory,evalInfo = runfn()
        retval = tuple(trainHistory, evalInfo)
        return retval

    print(f"Running {numRuns} experiments")
    with Pool(5) as p:
        results = p.map(worker, range(numRuns))

    train_stats = {}
    eval_stats = defaultdict(list)
    for trainHist, evalInfo in results:
        for key in trainHist.keys():
            data = np.array(trainHist[key])
            if key not in train_stats:
                train_stats[key] = {
                        'min': data, 'max': data,
                        'E[x^2]': data**2/numRuns, 
                        'E[x]': data/numRuns
                    }
            else:
                current = train_stats[key]
                updated_entry = {
                    'min': np.where(current['min'] < data, current['min'], data),
                    'max': np.where(current['max'] > data, current['min'], data),
                    'E[x]': current['E[x]'] + data/numRuns,
                    'E[x^2]': current['x^2'] + data**2/numRuns
                }
                train_stats[key] = updated_entry
        
        for key in evalInfo.keys():
            data = list(evalInfo[key])
            eval_stats[key].extend(data)
    

    # compute std
    for key in train_stats.keys():
        mean = train_stats[key].pop('E[x]')
        Ex2 = train_stats[key].pop('E[x^2]') 
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
            mean = data['mean']
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
