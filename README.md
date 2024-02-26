# Source Code for Offline Policy Evaluation and Optimization under Confounding

This folder contains the source code for the AISTATS 2024 paper "Offline Policy Evaluation and Optimization under Confounding".
The folder is structured as follows:
* `core/` contains utilities, helper classes and functions generously provided by David Bruns-Smith as part of the source code for his paper "Model-Free and Model-Based Policy Evaluation when Causality is Uncertain".
* `mcmix/` contains the source code for the global confounders portion of our paper. 
    * The `subspace.py`, `clustering.py`, `emalg.py`, and `helpers.py` files were obtained from the source code for "Learning Mixtures of Markov Chains and MDPs" by Kausik et. al. 
    * The folder `sepsisSimDiabetes/` contains the sepsis simulator of Oberst and Sontag, "Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models". `mdptoolboxSec/` and `cf/` contain code provided by them necessary to obtain the files in `data/`.
    * The `data/` folder contains (1) the sepsis simulator's transition matrix in `diab_txr_mats-replication.pkl`, (2) the epsilon-greedy behavior policy in `sepsisPol.npy`. The former can be re-obtained by running the notebook `learn_mdp_parameters.ipynb`, and the latter can be re-obtained by running `behavior_policy.ipynb`.
    * The main experiment for this portion of the paper can be reproduced by running `sepsisOPELarge.ipynb`.
* `COPE/` contains the source code for the history-independent confounders portion of the paper. `histIndep.ipynb` is self-contained and contains the main experiment for this portion of the paper.
