**Steps to run**

1 . _Create env_

`$ conda create -n qiskit-new python=3.11`

2. _Clean old versions of dependencies if you are using existing environment, otherwise skip to Step 3_
`$ pip uninstall -r requirements.txt`

3. _Install Dependencies_

`$ pip install -r requirements.txt`

4. _View help for usage of `tsp_problems.py`_
     - `$ python tsp_problems.py --help`
```
usage: tsp_problems.py [-h] [--nodes NODES] [--save-graph] [--real] [--algo {1,2,3}] [--spsa | --cobyla] [--maxiter MAXITER]

Solve TSP using QAOA and optionally save the graph.

options:
  -h, --help         show this help message and exit
  --nodes NODES      The number of cities(an integer > 2)
  --save-graph       Save the graph to a PNG file instead of displaying it
  --real             Run on IBM's real QPU
  --algo {1,2,3}     Algorithms to run (1=>QAOA only, 2=>Bruteforce only, 3=>Both)
  --spsa             Use the SPSA optimizer
  --cobyla           Use the COBYLA optimizer
```


5. _Two approaches to solve the TSP_

   - **Brute Force**
     - `$ python tsp_problems.py --nodes 10 --algo 2`  

   - **QAOA with qasm_simulator**
     - `$ python tsp_problems.py --nodes 3 --maxiter 3 --cobyla` for 3 nodes, using COBYLA optimizer max iterations=3
     - `$ python tsp_problems.py --nodes 3 --maxiter 5 --spsa` for 4 nodes, using SPSA optimizer max iterations=5

   - **QAOA with real Quantum Machine in IBM Cloud**
     - `python tsp_problems.py --nodes 10 --maxiter 5 --cobyla --real`