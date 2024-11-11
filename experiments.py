from tsp_problems import run_experiment
from tsp_problems import approxRatios
from tsp_problems import Algo
import penalty

if __name__ == "__main__":

    for maxiter in range(3,4):
        for k in range(1):
            run_experiment(5, optimizer_choice='COBYLA', optimizer_maxiter=maxiter, use_simulator=True, save_graph=False, algorithms=Algo.QAOA)
            penalty.incConst()
        penalty.reset()
        penalty.toggleOp()
        for k in range(1):
            run_experiment(5, optimizer_choice='COBYLA', optimizer_maxiter=maxiter, use_simulator=True, save_graph=False, algorithms=Algo.QAOA)
            penalty.incConst()

bestApproxRatio = min(approxRatios.items(), key=lambda x: x[1])[0]
print(approxRatios)
print(bestApproxRatio)
for item in approxRatios:
    if(item[1]==bestApproxRatio):
        print(f"Got best approx ratio {bestApproxRatio} for {item[0]}")