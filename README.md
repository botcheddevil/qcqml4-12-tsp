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


**License**

Contributions by Anshuk Kumar, including `compare.py`, are licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for the attribution to retain and the scope. Contributions by other authors remain under their own terms.
