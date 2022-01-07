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
    result = {'trainHist':trainHist, 'evalInfo':evalInfo, 'runId':runId}
    print('finished run:', runId, flush=True)
    # save stuff
    if savedir:
        save_as_json(f"{savedir}/data_run_{runId}.json", result) 
    return result


def save_as_json(fname, _object):
    try:
        with open(fname, 'w') as f: 
            json.dump(_object, f)
    except Exception as e:
        print(e)
        pass


def updateStats(hist:dict, runId:int, stats:dict, numRuns:int):
    for key in hist:
        if key == 'episode': continue
        if type(hist[key]) == dict:
            if key not in stats: stats[key] = {}
            updateStats(hist[key], runId, stats[key], numRuns) 
        elif type(hist[key]) in [np.ndarray, list]:
            try:
                data = np.asfarray(hist[key])
                if len(data) == 0: continue
                if key not in stats:
                    stats[key] = {
                        'episode': hist['episode'],
                        'min':data, 'max':data, 'mean':data/numRuns, 
                        'points':list(zip(hist['episode'], data)), 
                        'trace': [runId]*len(data),
                        'plotable':True
                    }
                else:
                    stats[key]['points'].extend(list(zip(hist['episode'], data)))
                    stats[key]['trace'].extend([runId]*len(data))
                    stats[key]['min'] = np.where(stats[key]['min'] < data, stats[key]['min'], data)
                    stats[key]['max'] = np.where(stats[key]['max'] > data, stats[key]['max'], data)
                    stats[key]['mean']= stats[key]['mean'] + data/numRuns
            except:
                pass

            
def get_stats(results):
    numRuns = len(results)
    print(f'accumulating {numRuns} results')
    train_stats = {}
    eval_stats = defaultdict(list)
    while results:
        result = results.pop(0)
        trainHist, evalInfo, runId = result['trainHist'], result['evalInfo'], result['runId']
        updateStats(trainHist, runId, train_stats, numRuns)
        
        for key in evalInfo.keys():
            data = list(evalInfo[key])
            eval_stats['trace'].extend([runId]*len(data))
            eval_stats[key].extend(data)
    
    return train_stats, eval_stats


def make_plots(stats:dict, movingavgon=[], savedir=None, show=False, title=''):
    for key in stats.keys():
        if type(stats[key]) is dict and 'plotable' not in stats[key]:
            make_plots(stats[key], movingavgon, savedir, show, f'{title}-{key}')
        else:
            data = stats[key]
            fn = movingAverage if key in movingavgon else lambda x:x
            if 'mean' in data:
                plt.plot(data['episode'], fn(data['mean']))
            if ('min' in data) and ('max' in data):
                plt.fill_between(data['episode'], fn(data['min']), fn(data['max']), alpha=0.2)
            if 'points' in data:
                xy = [*zip(*data['points'])]
                c =  data['trace'] if 'trace' in data else None
                plt.scatter(x=xy[0], y=xy[1], c=c, s=0.1)
            plt.xlabel('episode')
            plt.ylabel(key)
            if title: plt.title(f'{title}-{key}')
            if savedir: plt.savefig(f'{savedir}/trainplot_{title}-{key}.svg')
            if show: plt.show()
            plt.close()


def run_big_experiment(runfn, numRuns:int=1, numProcesses=4,
                    movingavgon:Union[list,None]=['reward'],
                    plot=True,savedir=None,show_plots=True):    
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

    # plot stuff
    if plot:
        make_plots(train_stats, movingavgon, savedir, show_plots, title=runfn.__name__)
    
    return train_stats, eval_stats
