from __future__ import annotations

import argparse
import math
import os
import os.path as osp

def arg_parse(argv=None):
    parser = argparse.ArgumentParser(description="IEC-GOOD self-supervised pretraining and supervised OOD fine-tuning")

    dataset_group = parser.add_argument_group("Dataset / protocol")
    dataset_group.add_argument("--DS", choices=("higraph", "bcg", "malnet-tiny", "call-graph"), default="call-graph", help="Benchmark dataset.")
    dataset_group.add_argument("--root", default=os.environ.get("IEC_GOOD_DATA_ROOT", "./data"), help="Root directory containing benchmark data.")
    dataset_group.add_argument("--bias", type=float, choices=(0.33, 0.66, 0.90), default=0.33, help="Graph-size OOD setting for MalNet-Tiny / Call-Graph.")
    dataset_group.add_argument("--malnet-data-dir", default=None, help="Optional MalNet-Tiny data parent directory.")
    dataset_group.add_argument("--malnet-split-dir", default=None, help="Optional explicit generated MalNet-Tiny split directory.")
    dataset_group.add_argument("--callgraph-data-dir", default=None, help="Optional Call-Graph dataset directory.")
    dataset_group.add_argument("--callgraph-split-dir", default=None, help="Optional explicit generated Call-Graph split directory.")
    dataset_group.add_argument("--callgraph-ood-dir", default=None, help="Optional parent directory containing generated Call-Graph OOD splits.")
    dataset_group.add_argument("--bcg-split-mode", choices=("intra_type", "cross_type"), default="cross_type", help="BCG Family-OOD evaluation protocol.")
    dataset_group.add_argument("--bcg-split-manifest-dir", default=None, help="Optional directory used to cache/reuse BCG split manifests.")

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    runtime_group.add_argument("--device", type=int, default=0, help="CUDA device index.")
    runtime_group.add_argument("--seed", type=int, default=0, help="Experiment random seed.")
    return parser.parse_args(argv)


def prepare_dataset_args(args, script_dir):
    del script_dir
    args.dataset = args.DS
    args.root = osp.abspath(osp.expanduser(args.root))

    if not args.malnet_data_dir:
        args.malnet_data_dir = args.root
    args.malnet_data_dir = osp.abspath(osp.expanduser(args.malnet_data_dir))

    path_args = ("malnet_split_dir", "callgraph_data_dir", "callgraph_split_dir", "callgraph_ood_dir", "bcg_split_manifest_dir")
    for name in path_args:
        value = getattr(args, name, None)
        if value:
            setattr(args, name, osp.abspath(osp.expanduser(value)))
    return args


def validate_training_args(args):
    if args.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {args.num_workers}.")
    if args.device < 0:
        raise ValueError(f"device must be non-negative, got {args.device}.")
    if args.seed < 0:
        raise ValueError(f"seed must be non-negative, got {args.seed}.")

    bias = float(args.bias)
    if not math.isfinite(bias) or bias not in {0.33, 0.66, 0.90}:
        raise ValueError("bias must be one of 0.33, 0.66, or 0.90.")
    return args
