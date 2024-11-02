from tsp_problems import run_experiment



if __name__ == "__main__":
    for nodes in range(4, 8):
        for maxiter in range(1, 4):
            run_experiment(nodes, optimizer_choice='cobyla', optimizer_maxiter=maxiter, use_simulator=True, save_graph=True)
            run_experiment(nodes, optimizer_choice='spsa', optimizer_maxiter=maxiter, use_simulator=True, save_graph=True)
