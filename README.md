# Simple FINancial Agent-Based Model (SIM-FIN-ABM)
This repository contains simple financial markets agent-based model (abm) with fundamentalist, mean-reversion & momentum traders
While this is a work in progress, the model works. Feel free to copy or use the code under the terms of the licence under LICENCE.txt. 

# How to use the model
1. Download the repository to your system.
2. Run the model using the simulations.py file or one of the Jupyter notebooks. 

# Repository structure

* The simfinmodel.py file contains the main logic for the model
* The functions folder contains supporting functions
* The objects folder contains the agent and orderbook classes which are used in the model.

For clarity reasons, I tried to stick to a functional programming style as much as possible. Therefore, most of the action takes 
place in the simfinmodel.py file.
