import statistics
from tsp_problems import run_experiment
from tsp_problems import approxRatios
from tsp_problems import Algo
import penalty

if __name__ == "__main__":

    for maxiter in range(3,9):
        penalty.setConst(1)
        penalty.setOp('+')
        for k in range(10):
            for r in range(6):
                # Repeat same experiment to test the consistency of results
                run_experiment(5, optimizer_choice='COBYLA', optimizer_maxiter=maxiter, use_simulator=True, save_graph=False, algorithms=Algo.QAOA)
            penalty.incConst()

        penalty.setConst(1)
        penalty.setOp('x')
        for k in range(10):
            for r in range(6):
                # Repeat same experiment to test the consistency of results
                run_experiment(5, optimizer_choice='COBYLA', optimizer_maxiter=maxiter, use_simulator=True, save_graph=False, algorithms=Algo.QAOA)
            penalty.incConst()

    print(approxRatios)
    bestResult = min(approxRatios.items(), key=lambda x: statistics.mean(x[1]))
    print('Best Result = ', bestResult)
    bestApproxMean = statistics.mean(bestResult[1])
    print('Best Mean Approx Ratio = ', bestApproxMean)
    
    bestConfigs = []
    for item in approxRatios.items():
        meanApprox = statistics.mean(item[1])
        print('debug: item: ',item)
        print('debug: meanApprox: ',meanApprox)
        if bestApproxMean == meanApprox:
            bestConfigs.append(item[0])
        
    print(f"Got best mean approx ratio {bestApproxMean} for these configs:")
    print(bestConfigs)