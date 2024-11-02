from tsp_problems import run_experiment



if __name__ == "__main__":
    for maxiter in range(1, 3):
        run_experiment(4, optimizer_choice='cobyla', optimizer_maxiter=maxiter, use_simulator=False)
        run_experiment(4, optimizer_choice='spsa', optimizer_maxiter=maxiter, use_simulator=False)
