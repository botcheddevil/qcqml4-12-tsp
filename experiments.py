from tsp_problems import run_experiment


if __name__ == "__main__":
    for nodes in range(5,7): # 5 to 7 nodes
        for maxiter in range(1, 3): # maxiter=1 and maxiter=2
            run_experiment(nodes, optimizer_choice='COBYLA', optimizer_maxiter=maxiter, use_simulator=True, save_graph=True)
            run_experiment(nodes, optimizer_choice='SPSA', optimizer_maxiter=maxiter, use_simulator=True, save_graph=True)
