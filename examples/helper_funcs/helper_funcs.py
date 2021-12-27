import json
import os
from collections import defaultdict
from functools import partial
from time import perf_counter
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from DRLagents.utils.helper_funcs import movingAverage
from torch.multiprocessing import Pool


def make_worker(runId, runfn, savedir):
    # run experiment
    print('started run:', runId, flush=True)
    trainHist, evalInfo = runfn()
    print('finished run:', runId, flush=True)
    # save stuff
    if savedir:
        print(f'Saving results from run {runId}', flush=True)
        save_json(f"{savedir}/data_run_{runId}.json", 
            {'trainHist':trainHist, 'evalInfo':evalInfo, 'runId':runId}) 
    return trainHist, evalInfo, runId


def save_json(fname, _object):
    try:
        with open(fname, 'w') as f: 
            json.dump(_object, f)
    except Exception as e:
        print(e)
        pass


def make_plots(train_stats, eval_stats, savedir=None):
    for key in train_stats.keys():
        data = train_stats[key]
        mean = data['E[x]']
        xy = [*zip(*data['points'])]
        plt.plot(mean)
        plt.fill_between(range(len(mean)), data['min'], data['max'], alpha=0.2)
        plt.scatter(xy[0],xy[1],s=0.1,c=data['trace'],alpha=0.6)
        plt.xlabel('episode')
        plt.ylabel(key)
        if savedir: plt.savefig(f'{savedir}/trainplot_{key}.svg')
        plt.show()
    
    for key in eval_stats.keys():
        data = eval_stats[key]
        plt.boxplot(data)
        plt.xlabel('eval-run')
        plt.ylabel(key)
        if savedir: plt.savefig(f'{savedir}/evalplot_{key}.svg')
        plt.show()


# generates plotting data from the results
def get_stats(results):
    train_stats = {}
    eval_stats = defaultdict(list)
    numRuns = len(results)
    print(f'accumulating {numRuns} results')
    while results:
        trainHist, evalInfo, runId = results.pop()
        for key in trainHist.keys():
            data = np.array(trainHist[key])
            if key not in train_stats:
                train_stats[key] = {
                        'min': data, 'max': data,
                        'E[x]': data/numRuns,
                        'points': list(enumerate(data)),
                    }
            else:
                curr = train_stats[key]
                curr['points'].extend(list(enumerate(data)))
                updated_entry = {
                    'min': np.where(curr['min'] < data, curr['min'], data),
                    'max': np.where(curr['max'] > data, curr['max'], data),
                    'E[x]': curr['E[x]'] + data/numRuns,
                    'points': curr['points']
                }
                train_stats[key] = updated_entry
        for key in evalInfo.keys():
            data = list(evalInfo[key])
            eval_stats[key].extend(data)
  
    return train_stats, eval_stats


def run_big_experiment(runfn, numRuns:int=1, numProcesses=8,
                    movingavgon:Union[list,None]=['trainRewards'],
                    plot=True,savedir=None):    
    start_time = perf_counter()

    if savedir and not os.path.exists(savedir):
        print(f"making {savedir}")
        os.makedirs(savedir)

    # run the experiments
    print(f"Running {numRuns} experiments")
    with Pool(min(numProcesses,numRuns)) as p:
        worker = partial(make_worker, runfn=runfn, savedir=savedir)
        results = p.map(worker, range(numRuns))
    print(f"total-time taken {(perf_counter() - start_time)/60:.2f} minutes")

    train_stats, eval_stats = get_stats(results)

    # apply moving average
    if movingavgon is not None:
        for key in movingavgon:
            for mkey in train_stats[key].keys():
                if mkey == 'points': continue
                train_stats[key][mkey] = movingAverage(train_stats[key][mkey])

    # plot stuff
    if plot:
        make_plots(train_stats, eval_stats, savedir)
    
    return train_stats, eval_stats
