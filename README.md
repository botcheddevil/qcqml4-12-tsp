**Steps to run**

1 . _Create env_

`$ conda create -n qiskit-new python=3.9`


2. _Install Dependencies_

`$ pip install -r requirements.txt`


3. _Three approaches to solve the TSP_

   - **Brute Force**
     - `$ python brute.py`


   - **Djikstras Algorithm**
     - `$ python djikstras.py`


   - **QAQO with qasm_simulator**
     - `$ python qaoa.py` for 3 nodes
     - `$ python qaoa-4nodes.py` for 4 nodes
     - `$ python qaoa-5nodes.py` for 5 nodes
