from tsp_problems import run_experiment
from tsp_problems import Algo

if __name__ == "__main__":
    #DONE
    # run_experiment(4, optimizer_choice='COBYLA', optimizer_maxiter=3, use_simulator=False, save_graph=True, algorithms=Algo.QAOA) # 15s
    # run_experiment(5, optimizer_choice='SPSA', optimizer_maxiter=1, use_simulator=False, save_graph=True, algorithms=Algo.QAOA) # 2m 45s
    # run_experiment(5, optimizer_choice='COBYLA', optimizer_maxiter=1, use_simulator=False, save_graph=True, algorithms=Algo.QAOA) # 2m 50s
    
    # IN PROGRESS
    # run_experiment(5, optimizer_choice='SPSA', optimizer_maxiter=2, use_simulator=False, save_graph=True, algorithms=Algo.QAOA) # 4m 35s
    # run_experiment(5, optimizer_choice='COBYLA', optimizer_maxiter=2, use_simulator=False, save_graph=True, algorithms=Algo.QAOA) # 5m

    # run_experiment(5, optimizer_choice='SPSA', optimizer_maxiter=3, use_simulator=False, save_graph=True, algorithms=Algo.QAOA) # 7m 45s
    # run_experiment(5, optimizer_choice='COBYLA', optimizer_maxiter=3, use_simulator=False, save_graph=True, algorithms=Algo.QAOA) # 8m

    # NEXT TO RUN
    # run_experiment(6, optimizer_choice='SPSA', optimizer_maxiter=1, use_simulator=False, save_graph=True, algorithms=Algo.QAOA) # 2m 30s
    # run_experiment(6, optimizer_choice='COBYLA', optimizer_maxiter=1, use_simulator=False, save_graph=True, algorithms=Algo.QAOA) # 2m 40s
    
    # run_experiment(6, optimizer_choice='SPSA', optimizer_maxiter=2, use_simulator=False, save_graph=True, algorithms=Algo.QAOA) # 4m 30s
    # run_experiment(6, optimizer_choice='COBYLA', optimizer_maxiter=2, use_simulator=False, save_graph=True, algorithms=Algo.QAOA) # 4m 45s