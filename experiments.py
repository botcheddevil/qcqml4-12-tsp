from tsp_problems import run_experiment
from tsp_problems import Algo

if __name__ == "__main__":
    run_experiment(10, optimizer_choice='COBYLA', optimizer_maxiter=5, use_simulator=False, save_graph=True, algorithms=Algo.BOTH)
    