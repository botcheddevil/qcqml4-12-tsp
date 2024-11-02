from tsp_problems import run_experiment
debug = False


if __name__ == "__main__":
    for maxiter in range(1, 10):
        run_experiment(4, optimizer_choice='cobyla', optimizer_maxiter=maxiter)
        run_experiment(4, optimizer_choice='spsa', optimizer_maxiter=maxiter)
