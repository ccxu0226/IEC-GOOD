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
from torch_geometric.datasets import MalNetTiny


SPLIT_STATISTIC = "num_nodes"
SPLIT_DIRECTION = "small_to_large"
SPLIT_VAL_MODE = "source"
TRAIN_PER_CLASS = 700
VAL_PER_CLASS = 100
TEST_PER_CLASS = 200


@dataclass(frozen=True)
class SplitConfig:
    bias: float
    seed: int
    statistic: str
    direction: str
    val_mode: str
    train_per_class: int
    val_per_class: int
    test_per_class: int
    save_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create the IEC-GOOD MalNet-Tiny graph-size OOD split. The released split uses num_nodes, small_to_large, source validation and a 700/100/200 per-class split; only Bias is varied.')
    parser.add_argument('--bias', type=float, choices=(0.33, 0.66, 0.9), default=0.33, help='Graph-size OOD bias used in the paper.')
    parser.add_argument('--seed', type=int, default=42, help='Split random seed.')
    parser.add_argument('--data-dir', type=Path, default=Path('../data'), help='Parent directory for MalNet-Tiny and generated splits.')
    parser.add_argument('--save-mode', choices=('indices', 'materialized', 'both'), default='indices', help='How to store the generated split.')
    parser.add_argument('--output-name', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def average_tied_percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        raise ValueError("Cannot rank an empty array.")
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


def graph_statistic(data, statistic: str) -> float:
    num_nodes = int(data.num_nodes)
    num_edges = int(data.num_edges)

    if statistic == "num_nodes":
        return float(math.log1p(num_nodes))
    if statistic == "num_edges":
        return float(math.log1p(num_edges))
    if statistic == "avg_degree":

        return float(math.log1p(num_edges / max(num_nodes, 1)))

    raise ValueError(f"Unsupported statistic: {statistic}")


def infer_label_names(dataset: MalNetTiny) -> Dict[int, str]:
    fallback = {
        int(y): f"class_{int(y)}"
        for y in sorted({int(dataset[i].y.view(-1)[0]) for i in range(len(dataset))})
    }

    try:
        split_dir = Path(dataset.raw_paths[1])
        y_map: Dict[str, int] = {}

        for split in ("train", "val", "test"):
            split_file = split_dir / f"{split}.txt"
            lines = [
                line.strip()
                for line in split_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for filename in lines:
                malware_type = filename.split("/")[0]
                if malware_type not in y_map:
                    y_map[malware_type] = len(y_map)

        inverse = {label: name for name, label in y_map.items()}
        if set(inverse) == set(fallback):
            return inverse
    except (OSError, AttributeError, IndexError):
        pass

    return fallback


def validate_requested_counts(
    class_counts: Mapping[int, int],
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
) -> None:
    requested = train_per_class + val_per_class + test_per_class
    if min(train_per_class, val_per_class, test_per_class) <= 0:
        raise ValueError("Train, validation, and test counts must all be positive.")

    mismatches = {
        label: count
        for label, count in class_counts.items()
        if count != requested
    }
    if mismatches:
        details = ", ".join(
            f"class {label}: found {count}, requested total {requested}"
            for label, count in sorted(mismatches.items())
        )
        raise ValueError(
            "The requested per-class counts do not match the dataset. "
            f"{details}. Adjust --train-per-class, --val-per-class, and "
            "--test-per-class."
        )


def build_split(
    dataset: MalNetTiny,
    config: SplitConfig,
) -> Tuple[Dict[str, List[int]], Dict[int, np.ndarray]]:
    labels = np.asarray(
        [int(dataset[i].y.view(-1)[0]) for i in range(len(dataset))],
        dtype=np.int64,
    )
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_counts = {
        int(label): int(count)
        for label, count in zip(unique_labels, counts)
    }

    validate_requested_counts(
        class_counts=class_counts,
        train_per_class=config.train_per_class,
        val_per_class=config.val_per_class,
        test_per_class=config.test_per_class,
    )

    split_indices: Dict[str, List[int]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    score_by_class: Dict[int, np.ndarray] = {}

    seed_sequence = np.random.SeedSequence(config.seed)
    class_seed_sequences = seed_sequence.spawn(len(unique_labels))

    for label, class_seed in zip(unique_labels, class_seed_sequences):
        label_int = int(label)
        class_indices = np.flatnonzero(labels == label).astype(np.int64)
        class_rng = np.random.default_rng(class_seed)

        values = np.asarray(
            [
                graph_statistic(dataset[int(index)], config.statistic)
                for index in class_indices
            ],
            dtype=np.float64,
        )
        ranks = average_tied_percentile_rank(values)

        if config.direction == "large_to_small":
            ranks = 1.0 - ranks

        random_component = class_rng.random(len(class_indices))
        domain_score = (
            config.bias * ranks
            + (1.0 - config.bias) * random_component
        )

        domain_score = (
            domain_score
            + np.finfo(np.float64).eps * random_component
        )

        order = np.argsort(domain_score, kind="stable")
        ordered_indices = class_indices[order]

        source_count = config.train_per_class + config.val_per_class
        source_pool = ordered_indices[:source_count].copy()
        test_indices = ordered_indices[
            source_count : source_count + config.test_per_class
        ].copy()

        if config.val_mode == "source":
            class_rng.shuffle(source_pool)
            train_indices = source_pool[: config.train_per_class]
            val_indices = source_pool[
                config.train_per_class :
                config.train_per_class + config.val_per_class
            ]
        else:
            train_indices = ordered_indices[: config.train_per_class]
            val_indices = ordered_indices[
                config.train_per_class :
                config.train_per_class + config.val_per_class
            ]

        split_indices["train"].extend(map(int, train_indices))
        split_indices["val"].extend(map(int, val_indices))
        split_indices["test"].extend(map(int, test_indices))
        score_by_class[label_int] = domain_score

    final_rng = np.random.default_rng(config.seed + 1_000_003)
    for split_name in split_indices:
        final_rng.shuffle(split_indices[split_name])

    verify_split(dataset, split_indices, config)
    return split_indices, score_by_class


def verify_split(
    dataset: MalNetTiny,
    split_indices: Mapping[str, Sequence[int]],
    config: SplitConfig,
) -> None:
    split_sets = {
        name: set(map(int, indices))
        for name, indices in split_indices.items()
    }

    if split_sets["train"] & split_sets["val"]:
        raise RuntimeError("Train and validation splits overlap.")
    if split_sets["train"] & split_sets["test"]:
        raise RuntimeError("Train and test splits overlap.")
    if split_sets["val"] & split_sets["test"]:
        raise RuntimeError("Validation and test splits overlap.")

    union = split_sets["train"] | split_sets["val"] | split_sets["test"]
    if len(union) != len(dataset):
        raise RuntimeError(
            f"Split covers {len(union)} graphs, but dataset has {len(dataset)}."
        )

    expected = {
        "train": config.train_per_class,
        "val": config.val_per_class,
        "test": config.test_per_class,
    }
    for split_name, indices in split_indices.items():
        labels = [
            int(dataset[int(index)].y.view(-1)[0])
            for index in indices
        ]
        unique, counts = np.unique(labels, return_counts=True)
        actual = dict(zip(map(int, unique), map(int, counts)))
        for label in sorted(actual):
            if actual[label] != expected[split_name]:
                raise RuntimeError(
                    f"{split_name}: class {label} has {actual[label]} graphs; "
                    f"expected {expected[split_name]}."
                )


def summarize_split(
    dataset: MalNetTiny,
    split_indices: Mapping[str, Sequence[int]],
    label_names: Mapping[int, str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for split_name in ("train", "val", "test"):
        indices = list(map(int, split_indices[split_name]))
        labels = np.asarray(
            [int(dataset[i].y.view(-1)[0]) for i in indices],
            dtype=np.int64,
        )

        for label in sorted(np.unique(labels)):
            class_indices = [
                index
                for index, current_label in zip(indices, labels)
                if int(current_label) == int(label)
            ]

            nodes = np.asarray(
                [int(dataset[i].num_nodes) for i in class_indices],
                dtype=np.float64,
            )
            edges = np.asarray(
                [int(dataset[i].num_edges) for i in class_indices],
                dtype=np.float64,
            )
            avg_degree = edges / np.maximum(nodes, 1.0)

            rows.append(
                {
                    "split": split_name,
                    "label": int(label),
                    "label_name": label_names.get(int(label), f"class_{int(label)}"),
                    "count": len(class_indices),
                    "num_nodes_mean": float(nodes.mean()),
                    "num_nodes_median": float(np.median(nodes)),
                    "num_nodes_q25": float(np.quantile(nodes, 0.25)),
                    "num_nodes_q75": float(np.quantile(nodes, 0.75)),
                    "num_nodes_min": int(nodes.min()),
                    "num_nodes_max": int(nodes.max()),
                    "num_edges_mean": float(edges.mean()),
                    "avg_degree_mean": float(avg_degree.mean()),
                }
            )

        nodes = np.asarray(
            [int(dataset[i].num_nodes) for i in indices],
            dtype=np.float64,
        )
        edges = np.asarray(
            [int(dataset[i].num_edges) for i in indices],
            dtype=np.float64,
        )
        avg_degree = edges / np.maximum(nodes, 1.0)
        rows.append(
            {
                "split": split_name,
                "label": "all",
                "label_name": "all",
                "count": len(indices),
                "num_nodes_mean": float(nodes.mean()),
                "num_nodes_median": float(np.median(nodes)),
                "num_nodes_q25": float(np.quantile(nodes, 0.25)),
                "num_nodes_q75": float(np.quantile(nodes, 0.75)),
                "num_nodes_min": int(nodes.min()),
                "num_nodes_max": int(nodes.max()),
                "num_edges_mean": float(edges.mean()),
                "avg_degree_mean": float(avg_degree.mean()),
            }
        )

    return rows


def write_statistics_csv(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError("No statistics rows to write.")

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_materialized_graphs(
    dataset: MalNetTiny,
    split_indices: Mapping[str, Sequence[int]],
    output_dir: Path,
) -> None:
    print(
        "Materializing graph lists. This duplicates graph data and may require "
        "substantial memory and disk space."
    )
    for split_name in ("train", "val", "test"):
        graph_list = [
            dataset[int(index)].clone()
            for index in split_indices[split_name]
        ]
        torch.save(graph_list, output_dir / f"{split_name}.pt")
        del graph_list


def format_bias_tag(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def default_output_name(config: SplitConfig) -> str:
    return (
        f"{config.statistic}_bias_{format_bias_tag(config.bias)}"
        f"_seed_{config.seed}_{config.direction}_{config.val_mode}_val"
    )


def print_summary(
    output_dir: Path,
    split_indices: Mapping[str, Sequence[int]],
    statistics: Sequence[Mapping[str, object]],
) -> None:
    print("\nSplit created successfully.")
    print(f"Output directory: {output_dir}")
    print(
        "Sizes: "
        + ", ".join(
            f"{name}={len(split_indices[name])}"
            for name in ("train", "val", "test")
        )
    )

    print("\nOverall structural statistics:")
    for row in statistics:
        if row["label"] == "all":
            print(
                f"  {str(row['split']):>5}: "
                f"nodes mean={float(row['num_nodes_mean']):.2f}, "
                f"median={float(row['num_nodes_median']):.2f}, "
                f"Q25={float(row['num_nodes_q25']):.2f}, "
                f"Q75={float(row['num_nodes_q75']):.2f}"
            )

    print("\nFiles:")
    for path in sorted(output_dir.iterdir()):
        print(f"  - {path.name}")


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.bias <= 1.0:
        raise ValueError("--bias must be in [0, 1].")

    set_global_seed(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    dataset_root = data_dir / "MalNetTiny"
    output_root = data_dir / "MalNetTiny_OOD"

    config = SplitConfig(
        bias=float(args.bias),
        seed=int(args.seed),
        statistic=SPLIT_STATISTIC,
        direction=SPLIT_DIRECTION,
        val_mode=SPLIT_VAL_MODE,
        train_per_class=TRAIN_PER_CLASS,
        val_per_class=VAL_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
        save_mode=str(args.save_mode),
    )

    output_name = args.output_name or default_output_name(config)
    output_dir = output_root / output_name

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite to replace it or choose --output-name."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading MalNet-Tiny into: {dataset_root}")
    dataset = MalNetTiny(root=str(dataset_root), split=None)
    print(f"Loaded {len(dataset)} graphs.")

    label_names = infer_label_names(dataset)
    split_indices, _ = build_split(dataset, config)
    statistics = summarize_split(dataset, split_indices, label_names)

    index_payload = {
        "format_version": 1,
        "dataset": "MalNetTiny",
        "dataset_root": str(dataset_root),
        "config": asdict(config),
        "label_names": label_names,
        "train": torch.tensor(split_indices["train"], dtype=torch.long),
        "val": torch.tensor(split_indices["val"], dtype=torch.long),
        "test": torch.tensor(split_indices["test"], dtype=torch.long),
    }
    torch.save(index_payload, output_dir / "split_indices.pt")

    metadata = {
        "format_version": 1,
        "dataset": "MalNetTiny",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "num_graphs": len(dataset),
        "num_classes": len(label_names),
        "label_names": {str(k): v for k, v in label_names.items()},
        "config": asdict(config),
        "split_sizes": {
            name: len(indices)
            for name, indices in split_indices.items()
        },
        "notes": [
            "Graphs, labels, nodes, and edges are not modified.",
            "Ranking and split assignment are performed independently within each class.",
            "The default source validation mode does not use target-domain samples.",
            "The original PyG dataset is stored under dataset_root.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_statistics_csv(statistics, output_dir / "split_statistics.csv")

    if config.save_mode in {"materialized", "both"}:
        save_materialized_graphs(dataset, split_indices, output_dir)

    loader_example = f"""from pathlib import Path
import torch
from torch_geometric.datasets import MalNetTiny


# Released paper split settings. Only Bias is varied in the graph-size OOD study.
SPLIT_STATISTIC = "num_nodes"
SPLIT_DIRECTION = "small_to_large"
SPLIT_VAL_MODE = "source"
TRAIN_PER_CLASS = 700
VAL_PER_CLASS = 100
TEST_PER_CLASS = 200

split_dir = Path(r"{output_dir}")
payload = torch.load(split_dir / "split_indices.pt", map_location="cpu")

dataset = MalNetTiny(root=payload["dataset_root"], split=None)

train_dataset = dataset[payload["train"].tolist()]
val_dataset = dataset[payload["val"].tolist()]
test_dataset = dataset[payload["test"].tolist()]

print(len(train_dataset), len(val_dataset), len(test_dataset))
"""
    (output_dir / "load_split_example.py").write_text(
        loader_example,
        encoding="utf-8",
    )

    print_summary(output_dir, split_indices, statistics)


if __name__ == "__main__":
    main()
