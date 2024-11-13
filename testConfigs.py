import numpy as np
from tsp_problems import run_experiment
from tsp_problems import approxRatios
from tsp_problems import Algo
import penalty

if __name__ == "__main__":
    testConfigs = [(4,'x',3), (4,'x',9), (5,'x',9), (6,'x',20), (7,'x',2)]

    for maxiter, op, penaltyMultiplier in testConfigs:
        for k in range(100):
            run_experiment(5, optimizer_choice='COBYLA', optimizer_maxiter=maxiter, use_simulator=True, save_graph=False, algorithms=Algo.QAOA)
            run_experiment(6, optimizer_choice='COBYLA', optimizer_maxiter=maxiter, use_simulator=True, save_graph=False, algorithms=Algo.QAOA)

print(approxRatios)
for config, ratioList in approxRatios.items():
    if(item[1]==bestApproxRatio):
        print(f"Got best approx ratio {bestApproxRatio} for {item[0]}")