from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class IECGOODConfig:
    num_gc_layers: int = 3
    hidden_dim: int = 32
    projection_dim: int = 96

    num_regions: int = 3
    min_region_size: int = 1
    max_region_coverage: float = 0.5
    region_temperature: float = 0.5

    semantic_clusters: int = 300
    context_clusters: int = 300
    cluster_refresh_interval: int = 20
    ema_momentum: float = 0.99

    joint_balance: float = 0.5
    environment_fusion_weight: float = 0.1

    view1_mask_ratio: float = 0.3
    view1_edge_drop_ratio: float = 0.2
    view2_mask_ratio: float = 0.2
    view2_edge_drop_ratio: float = 0.3
    evidence_budget_eta: float = 0.5

    coalition_weight: float = 0.8
    invariance_weight: float = 0.5
    uniformity_weight: float = 1.0
    uniformity_temperature: float = 0.9
    evidence_epsilon: float = 1e-8

    warmup_epochs: int = 20
    pretrain_epochs: int = 100
    grad_clip_norm: float = 0.0
    finetune_epochs: int = 50
    num_finetune_runs: int = 10

    batch_size: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5

    def as_dict(self):
        return asdict(self)


_DATASET_OVERRIDES = {
    "higraph": {
        "batch_size": 64,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "hidden_dim": 32,
        "num_regions": 5,
        "max_region_coverage": 0.4,
        "environment_fusion_weight": 0.5,
        "ema_momentum": 0.99,
        "coalition_weight": 1.0,
        "invariance_weight": 2.0,
    },
    "bcg": {
        "batch_size": 128,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "hidden_dim": 64,
        "num_regions": 5,
        "max_region_coverage": 0.3,
        "environment_fusion_weight": 1.0,
        "ema_momentum": 0.99,
    },
    "malnet-tiny": {
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "hidden_dim": 32,
        "num_regions": 3,
        "max_region_coverage": 0.3,
        "environment_fusion_weight": 0.5,
        "ema_momentum": 0.999,
        "coalition_weight": 1.0,
        "invariance_weight": 0.5,
    },
    "call-graph": {
        "batch_size": 256,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "hidden_dim": 32,
        "num_regions": 3,
        "max_region_coverage": 0.2,
        "environment_fusion_weight": 0.25,
        "ema_momentum": 0.99,
        "coalition_weight": 0.5,
        "invariance_weight": 0.5,
    },
}

_DATASET_ALIASES = {
    "callgraph": "call-graph",
    "callgraph-ood": "call-graph",
    "callgraph_ood": "call-graph",
    "malnettiny": "malnet-tiny",
    "malnet": "malnet-tiny",
    "mal-net": "malnet-tiny",
    "binary_malnet": "malnet-tiny",
    "binary-malnet": "malnet-tiny",
    "binarymalnet": "malnet-tiny",
    "malnet_binary": "malnet-tiny",
    "malnet-binary": "malnet-tiny",
}


def canonical_dataset_name(name: str) -> str:
    normalized = str(name).strip().lower()
    canonical = _DATASET_ALIASES.get(normalized, normalized)
    if canonical not in _DATASET_OVERRIDES:
        raise ValueError(f"Unsupported dataset: {name}. Expected one of: higraph, bcg, malnet-tiny, call-graph.")
    return canonical


def get_experiment_config(dataset_name: str) -> IECGOODConfig:
    canonical = canonical_dataset_name(dataset_name)
    values = IECGOODConfig().as_dict()
    values.update(_DATASET_OVERRIDES[canonical])
    return IECGOODConfig(**values)


HIGRAPH_FEATURE_MODE = "cfg_mean"

MALNET_SPLIT_STATISTIC = "num_nodes"
MALNET_SPLIT_DIRECTION = "small_to_large"
MALNET_VAL_MODE = "source"
MALNET_SPLIT_SEED = 42

CALLGRAPH_SPLIT_STATISTIC = "num_nodes"
CALLGRAPH_SPLIT_DIRECTION = "small_to_large"
CALLGRAPH_VAL_MODE = "source"
CALLGRAPH_SPLIT_SEED = 42

BCG_FEATURE_MODE = "ldp"
BCG_TIME_FIELD = "vt_year"
BCG_TRAIN_RATIO = 0.7
BCG_VAL_RATIO = 0.1
BCG_TEST_RATIO = 0.2
BCG_MIN_FAMILY_SAMPLES = 1
BCG_MIN_FAMILIES_PER_TYPE = 3
BCG_MIN_TYPE_SAMPLES = 1
