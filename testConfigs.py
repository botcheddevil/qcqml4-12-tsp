import itertools
import json
import os
import statistics
import numpy as np
from tsp_problems import run_experiment
from tsp_problems import approxRatios
from tsp_problems import Algo
import penalty


def save_results(results):
    filepath = "testConfigs_results2.json"
    with open(filepath, 'w') as file:
        json.dump(results, file)
        

if __name__ == "__main__":
    
    # testConfigs = [(4,'+',8), (4,'x',10), (5,'x',7), (6,'+',4), (6,'+',6), (6,'x',9)]
    testConfigs = [(4,'+',8),(6,'x',9)]
    for maxiter, penaltyOp, penaltyConst in testConfigs:
        penalty.setConst(penaltyConst)
        penalty.setOp(penaltyOp)
        for k in range(100):
            run_experiment(5, optimizer_choice='COBYLA', optimizer_maxiter=maxiter, use_simulator=True, save_graph=False, algorithms=Algo.QAOA)
            save_results(approxRatios)
            print(f"Next: iteration {k+1} for config ({maxiter}, {penaltyOp}, {penaltyConst}) :")

    print(approxRatios)

    bestResult = min(approxRatios.items(), key=lambda x: statistics.mean(x[1]))
    print('Best Result = ', bestResult)
    bestApproxMean = statistics.mean(bestResult[1])
    print('Best Mean Approx Ratio = ', bestApproxMean)
    
    # Mean analysis
    bestMeanConfigs = []
    for item in approxRatios.items():
        meanApprox = statistics.mean(item[1])
        # print('debug: item: ',item)
        # print('debug: meanApprox: ',meanApprox)
        if bestApproxMean == meanApprox:
            bestMeanConfigs.append(item[0])
        
    print(f"Got best mean approx ratio {bestApproxMean} for these configs:")
    print(bestMeanConfigs)

    # Frequency analysis
    goodFrequency = {}
    for item in approxRatios.items():
        goodCount = item[1].count(1.0) + item[1].count(1.125)
        if goodCount in goodFrequency:
            goodFrequency[goodCount].append(item[0])
        else:
            goodFrequency[goodCount] = [item[0]]

    # goodFrequencySorted = sorted(goodFrequency.items(), key=lambda x:x[1], reverse=True)
    
    frequenciesSet = sorted(goodFrequency.keys(), reverse=True)
    bestTwoFrequencies = frequenciesSet[:2]
    print("Best two frequencies are: ", bestTwoFrequencies)
    
    bestFreqConfigs = {
        bestTwoFrequencies[0]: goodFrequency[bestTwoFrequencies[0]],
        bestTwoFrequencies[1]: goodFrequency[bestTwoFrequencies[1]],
    }
    print("Best frequency configs are:")
    for x,y in bestFreqConfigs.items():
        print(x," => ",y)