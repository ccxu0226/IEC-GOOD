# IEC-GOOD

Review version of IEC-GOOD. This repository provides experimental configurations, data split scripts, and supporting implementation details. The full implementation will be made publicly available upon acceptance of the paper. Research hyperparameters are fixed in `config.py`; command-line arguments are limited to dataset paths, OOD settings, runtime options, and random seeds.


## Files

```text
main.py                     Training entry point
arguments.py                Runtime and dataset arguments
config.py                   Fixed experiment configuration
iec_good.py                 IEC-GOOD pretraining method
aug.py                      Evidence-aware graph augmentation
clustering.py               Semantic/context clustering
region_generator.py         Candidate subgraph generator
gin.py                      GIN encoder
eval.py                     Fine-tuning and evaluation
utils.py                    Dataset loading utilities
dataset_gen/call-graph.py    Call-Graph split generator
dataset_gen/malnet_tiny.py   MalNet-Tiny split generator
datasets/bcg_inter.py        BCG Family-OOD implementation
datasets/bcg_intra.py        BCG intra-type entry
datasets/higraph_dataset.py  HiGraph loader
```

## Environment

The paper uses Python 3.10, PyTorch 2.1.0, and CUDA 11.8.

```bash
pip install torch torch-geometric numpy scikit-learn pyarrow
```

Install PyTorch and PyG builds that match your local CUDA environment.

## Fixed experiment settings

The released settings are defined in `config.py`.

```text
semantic/context K-means = 300 / 300
cluster refresh interval = 20
lambda_jnt = 0.5
view-1 mask / edge-drop = 0.3 / 0.2
view-2 mask / edge-drop = 0.2 / 0.3
eta = 0.5
lambda_uni = 1.0
tau = 0.9

HiGraph:     lambda_coal = 1.0, lambda_inv = 2.0
BCG:         lambda_coal = 0.8, lambda_inv = 0.5
MalNet-Tiny: lambda_coal = 1.0, lambda_inv = 0.5
Call-Graph:  lambda_coal = 0.5, lambda_inv = 0.5
```

Dataset-specific batch size, learning rate, weight decay, encoder dimension, K, beta, lambda_env, and EMA momentum are also fixed in `config.py`.

## Call-Graph

Generate a graph-size OOD split:

```bash
python dataset_gen/call-graph.py --data-dir /path/to/callgraph --bias 0.33
```

Run IEC-GOOD:

```bash
python main.py --DS call-graph --callgraph-data-dir /path/to/callgraph --bias 0.33 --seed 0
```

## MalNet-Tiny

Generate a graph-size OOD split:

```bash
python dataset_gen/malnet_tiny.py --data-dir /path/to/data --bias 0.33
```

Run IEC-GOOD:

```bash
python main.py --DS malnet-tiny --malnet-data-dir /path/to/data --bias 0.33 --seed 0
```

For graph-size OOD experiments, Bias is one of `0.33`, `0.66`, or `0.90`.

## HiGraph

Paper split:

```text
Train:      2012-2018
Validation: 2019
Test:       2020, 2021
```

Run:

```bash
python main.py --DS higraph --root /path/to/data --seed 0
```

## BCG

Intra-Type Family-OOD:

```bash
python main.py --DS bcg --root /path/to/data --bcg-split-mode intra_type --seed 0
```

Cross-Type Family-OOD:

```bash
python main.py --DS bcg --root /path/to/data --bcg-split-mode cross_type --seed 0
```

## Command-line help

```bash     python main.py --help
```
