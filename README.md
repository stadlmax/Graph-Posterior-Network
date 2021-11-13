# Graph Posterior Network

This is the official code repository to the paper

[Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification](https://arxiv.org/pdf/2110.14012.pdf)<br>
Maximilian Stadler, Bertrand Charpentier, Simon Geisler, Daniel Zügner, Stephan Günnemann<br>
Conference on Neural Information Processing Systems (NeurIPS) 2021.

[[Paper](https://arxiv.org/pdf/2110.14012.pdf)]|[Video - coming soon]()]

![Diagram](uncertainty-axioms.png?raw=true "Diagram")


## Installation
We recommend running this code with its dependencies in a conda enviroment. To begin with, create a new conda environment with all the necessary dependencies assuming that you are in the **root directory** of this project:

```bash
conda env create -f gpn_environment.yml python==3.8 --force
```

Since the code is packaged, you will also have to setup the code accordingly. Assuming that you are in the **root directory** of this project, run:

```bash
conda activate gpn
pip3 install -e .
```


## Data
Since we rely on published datasets from the Torch-Geometric package, you don't have to download datasets manually. When you run experiments on supported datasets, those will be downloaded and placed in the corresponding data directories. You can run the following datasets
- CoraML
- CiteSeer
- PubMed
- AmazonPhotos
- AmazonComputers
- CoauthorCS
- CoauthorPhysics


## Running Experiments
The experimental setup builds upon Sacred and configuring experiments in `.yaml`files. We will provide configurations

- for vanilla node classification
- leave-out-class experiments
- experiments with isolated node perturbations
- experiments for feature shifts
- experiments for edge shifts

with a default fraction of perturbed nodes of 10%. We provide them for the smaller datasets (i.e. all except ogbn-arxiv) for hidden dimensions `H=10` and `H=16`. 

The main experimental script is `train_and_eval.py`. Assuming that you are in the **root directory** of this project for all further commands, you can run experiments with 


### Vanilla Node Classification
For the vanilla classification on the CoraML dataset with a hidden dimension of 16 or 10 respectively, run

```bash
python3 train_and_eval.py with configs/gpn/classification_gpn_16.yaml data.dataset=CoraML
python3 train_and_eval.py with configs/gpn/classification_gpn_10.yaml data.dataset=CoraML

```

If you have GPU-devices availale on your system, experiments will run on device `0` on default. If no CUDA-devices can be found, the code will revert back to running only on CPUs. Runs will produce assets per default. Also note that for running experiments for graphs under perturbations, you will have to run the corresponding vanilla classification experiment first.


### Options for Feature Shifts
We consider random features from Unit Gaussian Distribution (`normal`) and from a Bernoulli Distribution (`bernoulli_0.5`). When using the configuration `ood_features`, you can change those settings (key `ood_perturbation_type`) in the command line together with the fraction of perturbed nodes (key `ood_budget_per_graph`) or in the corresponding configurations files, for example as

```bash
python3 train_and_eval.py with configs/gpn/ood_features_gpn_16.yaml data.dataset=CoraML data.ood_perturbation_type=normal data.ood_budget_per_graph=0.025
python3 train_and_eval.py with configs/gpn/ood_features_gpn_16.yaml data.dataset=CoraML data.ood_perturbation_type=bernoulli_0.5 data.ood_budget_per_graph=0.025
```

For experiments considering perturbations in an isolated fashion, this applies accordingly but without the fraction of perturbed nodes, e.g. 

```bash
python3 train_and_eval.py with configs/gpn/ood_isolated_gpn_16.yaml data.dataset=CoraML data.ood_perturbation_type=normal
python3 train_and_eval.py with configs/gpn/ood_isolated_gpn_16.yaml data.dataset=CoraML data.ood_perturbation_type=bernoulli_0.5
```


### Options for Edge Shifts
We consider random edge perturbations and the global and untargeted DICE attack. Those attacks can be set with the key `ood_type` which can be either set to `random_attack_dice` or `random_edge_perturbations`. As above, those settings can be changed in the command line or in the corresponding configuration files. While the key `ood_budget_per_graph` refers to the fraction of perturbed nodes in the paragraph above, it describes the fraction of perturbed edges in this case.

```bash
python3 train_and_eval.py with configs/gpn/ood_features_gpn_16.yaml data.dataset=CoraML data.ood_type=random_attack_dice data.ood_budget_per_graph=0.025
python3 train_and_eval.py with configs/gpn/ood_features_gpn_16.yaml data.dataset=CoraML data.ood_type=random_edge_perturbations data.ood_budget_per_graph=0.025
```


### Further Options
With the settings above, you can reproduce our experimental results. If you want to change different architectural settings, simply change the corresponding keys in the configuration files with most of them being self-explanatory. 


## Structure
If you want to have a detailed look at our code, we give a brief overview of our code structure.

- `configs`: directory for model configurations
- `data`: directory for datasets
- `gpn`: source code
    - `gpn.data`: code related to loading datasets and creating ID and OOD datasets
    - `gpn.distributions`: code related to custom distributions similar to torch.distributions
    - `experiments`: main routines for running experiments, i.e. loading configs, setting up datasets and models, training and evaluation
    - `gpn.layers`: custom layers
    - `gpn.models`: implementation of reference models and Graph Posterior Network (+ablated models)
    - `gpn.nn`: training related utilities like losses, metrics, or training engines
    - `gpn.utils`: general utility code
- `saved_experiments`: directory for saved models
- `train_and_eval.py`: main script for training & evaluation
- `gpn_qualitative_evaluation.ipynb`: jupyter notebook which evaluates the results from Graph Posterior Network in a qualitative fashion

Note that we provide the implementations of most of our used reference models. Our main Graph Posterior Network model can be found in `gpn.models.gpn_base.py`. Ablated models can be found in a similar fashion, i.e. PostNet in `gpn.models.gpn_postnet.py`, PostNet+diffusion in `gpn.models.gpn_postnet_diff.py` and the model diffusiong log-beta scores in `gpn.models.gpn_log_beta.py`.


We provide all basic configurations for reference models in `configs/reference`. Note that some models have dependencies with others, e.g. running `classification_gcn_dropout.yaml` or `classification_gcn_energy.yaml` would require training the underlying GCN first by running `classification_gcn.yaml` first, running `classification_gcn_ensemble.yaml` would require training 10 GCNs first with `init_no` in 1...10, and running `classification_sgcn.yaml` (GKDE-GCN) would require training the teacher-GCN first by running `classification_gcn.yaml` and computing the kernel values by running `classification_gdk.yaml` first. 


## Cite

Please cite our paper if you use the model or this code in your own work.

```
@incollection{graph-postnet,
title={Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification},
author={Stadler, Maximilian and Charpentier, Bertrand and Geisler, Simon and Z{\"u}gner, Daniel and G{\"u}nnemann, Stephan},
booktitle = {Advances in Neural Information Processing Systems},
volume = {34},
publisher = {Curran Associates, Inc.},
year = {2021}
}
```
