from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.separate import separate


EXPECTED_NUM_GRAPHS = 1361
EXPECTED_FEATURE_DIM = 27
EXPECTED_CLASS_SIZES = {546, 815}


SPLIT_STATISTIC = "num_nodes"
SPLIT_DIRECTION = "small_to_large"
SPLIT_VAL_MODE = "source"


@dataclass(frozen=True)
class SplitConfig:
    bias: float
    seed: int
    statistic: str
    direction: str
    val_mode: str
    save_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create the IEC-GOOD Call-Graph graph-size OOD split. The released split uses num_nodes, small_to_large, source validation; only Bias is varied across 0.33/0.66/0.90.')
    parser.add_argument('--bias', type=float, choices=(0.33, 0.66, 0.9), default=0.33, help='Graph-size OOD bias used in the paper.')
    parser.add_argument('--seed', type=int, default=42, help='Split random seed.')
    parser.add_argument('--data-dir', type=Path, default=Path('../data/dataset-call-graph-blogpost-material-master'), help='Root directory of the Call-Graph dataset.')
    parser.add_argument('--save-mode', choices=('indices', 'materialized', 'both'), default='indices', help='How to store the generated split.')
    parser.add_argument('--output-name', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_callgraph_dataset(dataset_root: Path) -> List[Data]:
    processed_path = dataset_root / "processed" / "data.pt"
    if not processed_path.is_file():
        raise FileNotFoundError(f'Processed Call-Graph dataset not found: {processed_path}')

    print(f"Loading processed Call-Graph dataset: {processed_path}")
    payload = torch.load(processed_path, map_location="cpu")

    if not isinstance(payload, tuple) or len(payload) != 2:
        raise RuntimeError('Expected processed/data.pt to contain a PyG (data, slices) tuple.')
    data, slices = payload
    required_keys = {"x", "edge_index", "y"}

    if not required_keys.issubset(set(data.keys())):
        raise RuntimeError(f'Call-Graph data must contain {required_keys}; observed {set(data.keys())}.')
    if not required_keys.issubset(set(slices.keys())):
        raise RuntimeError(f'Call-Graph slices must contain {required_keys}; observed {set(slices.keys())}.')

    num_graphs = int(slices["y"].numel() - 1)
    if num_graphs != EXPECTED_NUM_GRAPHS:
        raise RuntimeError(f'Expected {EXPECTED_NUM_GRAPHS} Call-Graph graphs, observed {num_graphs}.')

    graphs: List[Data] = []

    for index in range(num_graphs):
        graph = separate(cls=data.__class__, batch=data, idx=index, slice_dict=slices, decrement=False)

        if graph.x is None:
            raise RuntimeError(f'Graph {index} does not contain node features.')
        if graph.x.dim() != 2:
            raise RuntimeError(f'Graph {index} has invalid x shape {tuple(graph.x.shape)}.')
        if graph.x.size(1) != EXPECTED_FEATURE_DIM:
            raise RuntimeError(f'Graph {index} has feature dimension {graph.x.size(1)}; expected {EXPECTED_FEATURE_DIM}.')
        if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
            raise RuntimeError(f'Graph {index} has invalid edge_index shape {tuple(graph.edge_index.shape)}.')

        graph.edge_index = graph.edge_index.long()
        graph.y = graph.y.view(-1).long()

        if graph.y.numel() != 1:
            raise RuntimeError(f'Graph {index} has {graph.y.numel()} labels; expected exactly one.')

        num_nodes = int(graph.x.size(0))
        graph.num_nodes = num_nodes
        graph.graph_id = torch.tensor([index], dtype=torch.long)

        if graph.edge_index.numel() > 0:
            minimum, maximum = int(graph.edge_index.min()), int(graph.edge_index.max())
            if minimum < 0 or maximum >= num_nodes:
                raise RuntimeError(f'Graph {index} has invalid edge indices [{minimum}, {maximum}] for {num_nodes} nodes.')

        graphs.append(graph)

    print(f"Loaded {len(graphs)} Call-Graph graphs.")
    return graphs


def infer_label_names(dataset: Sequence[Data]) -> Dict[int, str]:
    labels = np.asarray([int(dataset[i].y.view(-1)[0]) for i in range(len(dataset))], dtype=np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)

    if len(unique_labels) != 2:
        raise RuntimeError(f'Expected two Call-Graph classes, observed {len(unique_labels)}: {unique_labels.tolist()}.')

    class_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    if set(class_counts.values()) != EXPECTED_CLASS_SIZES:
        raise RuntimeError(f'Expected Call-Graph class sizes 546 and 815, observed {class_counts}.')

    label_names: Dict[int, str] = {}
    for label, count in class_counts.items():
        if count == 546: label_names[label] = "goodware"
        if count == 815:
            label_names[label] = 'malware'

    return label_names


def class_split_counts(class_size: int) -> Dict[str, int]:
    if class_size == 546:
        return {'train': 382, 'val': 55, 'test': 109}
    if class_size == 815:
        return {'train': 571, 'val': 81, 'test': 163}
    raise ValueError(f"Unsupported Call-Graph class size: {class_size}.")


def average_tied_percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n == 0:
        raise ValueError('Cannot rank an empty array.')
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0

    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        average_position = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_position / (n - 1)
        start = end

    return ranks


def graph_statistic(data: Data, statistic: str) -> float:
    num_nodes, num_edges = int(data.num_nodes), int(data.num_edges)
    if statistic == 'num_nodes':
        return float(math.log1p(num_nodes))
    if statistic == 'num_edges':
        return float(math.log1p(num_edges))
    if statistic == 'avg_degree':
        return float(math.log1p(num_edges / max(num_nodes, 1)))
    raise ValueError(f"Unsupported statistic: {statistic}")


def build_split(dataset: Sequence[Data], config: SplitConfig) -> Tuple[Dict[str, List[int]], Dict[int, np.ndarray]]:
    labels = np.asarray([int(dataset[i].y.view(-1)[0]) for i in range(len(dataset))], dtype=np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    split_indices: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    score_by_class: Dict[int, np.ndarray] = {}

    seed_sequence = np.random.SeedSequence(config.seed)
    class_seed_sequences = seed_sequence.spawn(len(unique_labels))

    for label, class_seed in zip(unique_labels, class_seed_sequences):
        label_int = int(label)
        class_indices = np.flatnonzero(labels == label).astype(np.int64)
        class_rng = np.random.default_rng(class_seed)
        split_counts = class_split_counts(class_counts[label_int])

        values = np.asarray([graph_statistic(dataset[int(index)], config.statistic) for index in class_indices], dtype=np.float64)
        ranks = average_tied_percentile_rank(values)

        if config.direction == 'large_to_small':
            ranks = 1.0 - ranks

        random_component = class_rng.random(len(class_indices))
        domain_score = config.bias * ranks + (1.0 - config.bias) * random_component


        order = np.lexsort((random_component, domain_score))
        ordered_indices = class_indices[order]

        source_count = split_counts["train"] + split_counts["val"]
        source_pool = ordered_indices[:source_count].copy()
        test_indices = ordered_indices[source_count:source_count + split_counts["test"]].copy()

        if config.val_mode == "source":
            class_rng.shuffle(source_pool)
            train_indices = source_pool[:split_counts["train"]]
            val_indices = source_pool[split_counts["train"]:split_counts["train"] + split_counts["val"]]
        else:
            train_indices = ordered_indices[:split_counts["train"]]
            val_indices = ordered_indices[split_counts["train"]:split_counts["train"] + split_counts["val"]]

        split_indices["train"].extend(map(int, train_indices))
        split_indices["val"].extend(map(int, val_indices))
        split_indices["test"].extend(map(int, test_indices))
        score_by_class[label_int] = domain_score

    final_rng = np.random.default_rng(config.seed + 1_000_003)
    for split_name in split_indices:
        final_rng.shuffle(split_indices[split_name])

    verify_split(dataset, split_indices)
    return split_indices, score_by_class


def verify_split(dataset: Sequence[Data], split_indices: Mapping[str, Sequence[int]]) -> None:
    split_sets = {name: set(map(int, indices)) for name, indices in split_indices.items()}

    if split_sets['train'] & split_sets['val']:
        raise RuntimeError('Train and validation splits overlap.')
    if split_sets['train'] & split_sets['test']:
        raise RuntimeError('Train and test splits overlap.')
    if split_sets['val'] & split_sets['test']:
        raise RuntimeError('Validation and test splits overlap.')

    union = split_sets["train"] | split_sets["val"] | split_sets["test"]
    if len(union) != len(dataset):
        raise RuntimeError(f'Split covers {len(union)} graphs, but dataset has {len(dataset)}.')

    labels = np.asarray([int(dataset[i].y.view(-1)[0]) for i in range(len(dataset))], dtype=np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    for split_name in ("train", "val", "test"):
        split_labels = [int(dataset[int(index)].y.view(-1)[0]) for index in split_indices[split_name]]
        unique, current_counts = np.unique(split_labels, return_counts=True)
        actual = {int(label): int(count) for label, count in zip(unique, current_counts)}

        for label in unique_labels:
            label_int = int(label)
            expected = class_split_counts(class_counts[label_int])[split_name]
            observed = actual.get(label_int, 0)
            if observed != expected:
                raise RuntimeError(f'{split_name}: class {label_int} has {observed} graphs; expected {expected}.')


def summarize_split(dataset: Sequence[Data], split_indices: Mapping[str, Sequence[int]], label_names: Mapping[int, str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for split_name in ("train", "val", "test"):
        indices = list(map(int, split_indices[split_name]))
        labels = np.asarray([int(dataset[i].y.view(-1)[0]) for i in indices], dtype=np.int64)

        for label in sorted(np.unique(labels)):
            class_indices = [index for index, current_label in zip(indices, labels) if int(current_label) == int(label)]
            rows.append(build_statistics_row(dataset, class_indices, split_name, int(label), label_names.get(int(label), f"class_{int(label)}")))

        rows.append(build_statistics_row(dataset, indices, split_name, "all", "all"))

    return rows


def build_statistics_row(dataset: Sequence[Data], indices: Sequence[int], split_name: str, label, label_name: str) -> Dict[str, object]:
    nodes = np.asarray([int(dataset[i].num_nodes) for i in indices], dtype=np.float64)
    edges = np.asarray([int(dataset[i].num_edges) for i in indices], dtype=np.float64)
    avg_degree = edges / np.maximum(nodes, 1.0)

    return {
        "split": split_name,
        "label": label,
        "label_name": label_name,
        "count": len(indices),
        "num_nodes_mean": float(nodes.mean()),
        "num_nodes_median": float(np.median(nodes)),
        "num_nodes_q25": float(np.quantile(nodes, 0.25)),
        "num_nodes_q75": float(np.quantile(nodes, 0.75)),
        "num_nodes_min": int(nodes.min()),
        "num_nodes_max": int(nodes.max()),
        "num_edges_mean": float(edges.mean()),
        "num_edges_median": float(np.median(edges)),
        "num_edges_min": int(edges.min()),
        "num_edges_max": int(edges.max()),
        "avg_degree_mean": float(avg_degree.mean()),
    }


def write_statistics_csv(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError('No statistics rows to write.')

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_materialized_graphs(dataset: Sequence[Data], split_indices: Mapping[str, Sequence[int]], output_dir: Path) -> None:
    print("Materializing graph lists. This duplicates graph data and may require additional disk space.")

    for split_name in ("train", "val", "test"):
        graph_list = [dataset[int(index)].clone() for index in split_indices[split_name]]
        torch.save(graph_list, output_dir / f"{split_name}.pt")
        del graph_list


def format_bias_tag(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def default_output_name(config: SplitConfig) -> str:
    return f"{config.statistic}_bias_{format_bias_tag(config.bias)}_seed_{config.seed}_{config.direction}_{config.val_mode}_val"


def print_dataset_summary(dataset: Sequence[Data], label_names: Mapping[int, str]) -> None:
    labels = np.asarray([int(data.y.view(-1)[0]) for data in dataset], dtype=np.int64)

    print("=" * 70)
    print("Call-Graph dataset")
    print(f"Graphs: {len(dataset)}")
    print(f"Input dimension: {dataset[0].x.size(1)}")

    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        nodes = np.asarray([int(dataset[int(index)].num_nodes) for index in indices], dtype=np.float64)
        edges = np.asarray([int(dataset[int(index)].num_edges) for index in indices], dtype=np.float64)
        print(f"Class {int(label)} ({label_names[int(label)]}): graphs={len(indices)} | nodes mean={nodes.mean():.2f} | nodes median={np.median(nodes):.2f} | nodes min={int(nodes.min())} | nodes max={int(nodes.max())} | edges mean={edges.mean():.2f}")

    print("=" * 70)


def print_summary(output_dir: Path, split_indices: Mapping[str, Sequence[int]], statistics: Sequence[Mapping[str, object]], label_names: Mapping[int, str], dataset: Sequence[Data]) -> None:
    print("\nSplit created successfully.")
    print(f"Output directory: {output_dir}")
    print("Sizes: " + ", ".join(f"{name}={len(split_indices[name])}" for name in ("train", "val", "test")))

    print("\nClass distribution:")

    for split_name in ("train", "val", "test"):
        labels = [int(dataset[int(index)].y.view(-1)[0]) for index in split_indices[split_name]]
        text = [f"{label_names[label]}={sum(current == label for current in labels)}" for label in sorted(label_names)]
        print(f"  {split_name:>5}: " + ", ".join(text))

    print("\nOverall structural statistics:")

    for row in statistics:
        if row["label"] == "all":
            print(f"  {str(row['split']):>5}: nodes mean={float(row['num_nodes_mean']):.2f}, median={float(row['num_nodes_median']):.2f}, Q25={float(row['num_nodes_q25']):.2f}, Q75={float(row['num_nodes_q75']):.2f}, min={int(row['num_nodes_min'])}, max={int(row['num_nodes_max'])}")

    print("\nFiles:")
    for path in sorted(output_dir.iterdir()):
        print(f'  - {path.name}')


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.bias <= 1.0:
        raise ValueError('--bias must be in [0, 1].')

    set_global_seed(args.seed)

    dataset_root = args.data_dir.expanduser().resolve()
    processed_path = dataset_root / "processed" / "data.pt"

    if not processed_path.is_file():
        raise FileNotFoundError(f'Call-Graph processed dataset not found: {processed_path}')

    output_root = dataset_root.parent / "CallGraph_OOD"
    config = SplitConfig(
        bias=float(args.bias),
        seed=int(args.seed),
        statistic=SPLIT_STATISTIC,
        direction=SPLIT_DIRECTION,
        val_mode=SPLIT_VAL_MODE,
        save_mode=str(args.save_mode),
    )
    output_name = args.output_name or default_output_name(config)
    output_dir = output_root / output_name

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f'Output directory already exists: {output_dir}\nUse --overwrite to replace it or choose --output-name.')
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)

    dataset = load_callgraph_dataset(dataset_root)
    label_names = infer_label_names(dataset)
    print_dataset_summary(dataset, label_names)

    split_indices, _ = build_split(dataset, config)
    statistics = summarize_split(dataset, split_indices, label_names)

    index_payload = {
        "format_version": 1,
        "dataset": "CallGraph",
        "dataset_root": str(dataset_root),
        "processed_file": str(processed_path),
        "config": asdict(config),
        "label_names": label_names,
        "train": torch.tensor(split_indices["train"], dtype=torch.long),
        "val": torch.tensor(split_indices["val"], dtype=torch.long),
        "test": torch.tensor(split_indices["test"], dtype=torch.long),
    }
    torch.save(index_payload, output_dir / "split_indices.pt")

    labels = np.asarray([int(data.y.view(-1)[0]) for data in dataset], dtype=np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_counts = {label_names[int(label)]: int(count) for label, count in zip(unique_labels, counts)}

    metadata = {
        "format_version": 1,
        "dataset": "CallGraph",
        "dataset_root": str(dataset_root),
        "processed_file": str(processed_path),
        "output_dir": str(output_dir),
        "num_graphs": len(dataset),
        "num_classes": len(label_names),
        "input_dim": EXPECTED_FEATURE_DIM,
        "label_names": {str(k): v for k, v in label_names.items()},
        "class_counts": class_counts,
        "config": asdict(config),
        "split_sizes": {name: len(indices) for name, indices in split_indices.items()},
        "notes": [
            "Graphs, labels, nodes, edges, and node-feature values are not modified.",
            "Node features are the original 27-dimensional opcode-count vectors.",
            "Ranking and split assignment are performed independently within each class.",
            "The source validation mode does not use target-domain samples.",
            "The original directed edge_index is preserved.",
            "The original processed PyG dataset is stored under dataset_root/processed/data.pt.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_statistics_csv(statistics, output_dir / "split_statistics.csv")

    if config.save_mode in {'materialized', 'both'}:
        save_materialized_graphs(dataset, split_indices, output_dir)

    loader_example = f'''from pathlib import Path
import torch
from torch_geometric.data.separate import separate

split_dir = Path(r"{output_dir}")
payload = torch.load(split_dir / "split_indices.pt", map_location="cpu")
data, slices = torch.load(payload["processed_file"], map_location="cpu")
num_graphs = int(slices["y"].numel() - 1)

dataset = []
for index in range(num_graphs):
    graph = separate(cls=data.__class__, batch=data, idx=index, slice_dict=slices, decrement=False)
    graph.edge_index = graph.edge_index.long()
    graph.y = graph.y.view(-1).long()
    graph.num_nodes = int(graph.x.size(0))
    dataset.append(graph)

train_dataset = [dataset[int(i)] for i in payload["train"]]
val_dataset = [dataset[int(i)] for i in payload["val"]]
test_dataset = [dataset[int(i)] for i in payload["test"]]

print(len(train_dataset), len(val_dataset), len(test_dataset))
'''
    (output_dir / "load_split_example.py").write_text(loader_example, encoding="utf-8")
    print_summary(output_dir, split_indices, statistics, label_names, dataset)


if __name__ == "__main__":
    main()
