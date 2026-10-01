import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from torch_geometric.datasets import MalNetTiny

from aug import ensure_node_features
from config import (
    BCG_FEATURE_MODE,
    BCG_MIN_FAMILIES_PER_TYPE,
    BCG_MIN_FAMILY_SAMPLES,
    BCG_MIN_TYPE_SAMPLES,
    BCG_TEST_RATIO,
    BCG_TIME_FIELD,
    BCG_TRAIN_RATIO,
    BCG_VAL_RATIO,
    CALLGRAPH_SPLIT_DIRECTION,
    CALLGRAPH_SPLIT_SEED,
    CALLGRAPH_SPLIT_STATISTIC,
    CALLGRAPH_VAL_MODE,
    HIGRAPH_FEATURE_MODE,
    MALNET_SPLIT_DIRECTION,
    MALNET_SPLIT_SEED,
    MALNET_SPLIT_STATISTIC,
    MALNET_VAL_MODE,
)
from datasets.higraph_dataset import HiGraphDataset
from datasets.bcg_inter import build_bcg_family_ood_datasets as build_bcg_inter_datasets
from datasets.bcg_intra import build_bcg_family_ood_datasets as build_bcg_intra_datasets

class UnlabeledGraphDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = ensure_node_features(self.dataset[int(index)].clone())
        if "y" in data:
            del data.y
        data.pretrain_id = torch.tensor([int(index)], dtype=torch.long)
        return data


class CallGraphDataset(Dataset):
    def __init__(self, graphs, indices):
        self.graphs = graphs
        self.indices = [int(index) for index in indices]
        self.input_dim = 27

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.graphs[self.indices[index]]


class BinaryMalNetDataset(Dataset):
    def __init__(self, dataset, benign_label):
        self.dataset = dataset
        self.benign_label = int(benign_label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index].clone()
        original_label = int(torch.as_tensor(data.y).view(-1)[0].item())
        data.y = torch.tensor([0 if original_label == self.benign_label else 1], dtype=torch.long)
        return data


@dataclass
class DatasetBundle:
    train_dataset: Any
    val_dataset: Any
    test_dataset: Any
    input_dim: int
    num_classes: int
    temporal_test_datasets: Any = None


def setup_seed(seed):
    seed = int(seed)
    if seed < 0:
        raise ValueError(f'seed must be non-negative, got {seed}.')

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def get_device(device_id):
    if int(device_id) < 0:
        raise ValueError(f'device_id must be non-negative, got {device_id}.')

    if torch.cuda.is_available():
        if int(device_id) >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {device_id} is unavailable; detected {torch.cuda.device_count()} CUDA devices.')
        return torch.device(f"cuda:{int(device_id)}")

    return torch.device("cpu")


def attach_graph_ids(dataset, start=0):
    result = []

    for index in range(len(dataset)):
        data = dataset[index]
        if not isinstance(data, Data):
            raise TypeError(f'Dataset item {index} must be a PyG Data object, got {type(data).__name__}.')

        data = data.clone()
        data.graph_id = torch.tensor([int(start) + index], dtype=torch.long)
        result.append(data)

    return result


def infer_num_classes(dataset):
    labels = []

    for index in range(len(dataset)):
        data = dataset[index]
        target = getattr(data, "y", None)
        if target is None:
            raise ValueError(f'Training graph {index} does not contain y.')

        target = torch.as_tensor(target).detach().view(-1)
        valid_target = target[torch.isfinite(target)]
        if valid_target.numel() > 0:
            labels.extend(valid_target.long().tolist())

    if not labels:
        raise ValueError('The training dataset contains no valid labels.')

    classes = torch.unique(torch.tensor(labels, dtype=torch.long), sorted=True)
    expected_classes = torch.arange(classes.numel(), dtype=torch.long)
    if not torch.equal(classes, expected_classes):
        raise ValueError(f'Training labels must be contiguous and start from zero. Observed classes: {classes.tolist()}.')

    return int(classes.numel())


def infer_input_dimension(dataset):
    if len(dataset) == 0:
        raise ValueError('Cannot infer input dimension from an empty dataset.')

    for index in range(len(dataset)):
        data = dataset[index]
        x = getattr(data, "x", None)

        if x is None:
            continue
        if x.dim() == 1:
            return 1
        if x.dim() != 2:
            raise ValueError(f'Graph {index} node features must have shape [N, F], got {x.shape}.')

        return int(x.size(1))

    return 1


def load_dataset_bundle(args):
    dataset_name = str(getattr(args, "dataset", args.DS))
    name = dataset_name.strip().lower()

    if name == "higraph":
        bundle = _load_higraph_bundle(args)
    elif name == "bcg":
        bundle = _load_bcg_bundle(args)
    elif name in {
        "malnet-tiny",
        "malnettiny",
        "malnet",
        "mal-net",
        "binary_malnet",
        "binary-malnet",
        "binarymalnet",
        "malnet_binary",
        "malnet-binary",
    }:
        bundle = _load_binary_malnet_bundle(args)
    elif name in {"callgraph", "call-graph", "callgraph_ood", "callgraph-ood"}:
        bundle = _load_callgraph_bundle(args)
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            "Expected one of: higraph, bcg, malnet-tiny, call-graph."
        )

    _validate_dataset_bundle(bundle)
    return bundle


def _resolve_higraph_root(args):
    root = os.path.abspath(os.path.expanduser(args.root))
    higraph_root = root if os.path.basename(os.path.normpath(root)).lower() == "higraph_data" else os.path.join(root, "HiGraph_data")
    if not os.path.isdir(higraph_root):
        raise FileNotFoundError(f'HiGraph directory was not found: {higraph_root}')
    return higraph_root


def _load_higraph_bundle(args):
    root = _resolve_higraph_root(args)
    feature_mode = HIGRAPH_FEATURE_MODE

    train = HiGraphDataset(higraph_root=root, years=range(2012, 2019), split_name="train_2012_2018", feature_mode=feature_mode)
    val = HiGraphDataset(higraph_root=root, years=[2019], split_name="val_2019", feature_mode=feature_mode)
    test_2020 = HiGraphDataset(higraph_root=root, years=[2020], split_name="test_2020", feature_mode=feature_mode)
    test_2021 = HiGraphDataset(higraph_root=root, years=[2021], split_name="test_2021", feature_mode=feature_mode)


    temporal_test_datasets = {"2020": test_2020, "2021": test_2021}
    test = ConcatDataset([test_2020, test_2021])

    print(f"HiGraph dataset root: {root}")
    print(f"HiGraph feature mode: {feature_mode}")
    print(f"HiGraph train years: 2012-2018 | graphs={len(train)}")
    print(f"HiGraph validation year: 2019 | graphs={len(val)}")
    print(f"HiGraph test year 2020 | graphs={len(test_2020)}")
    print(f"HiGraph test year 2021 | graphs={len(test_2021)}")
    print(f"HiGraph total temporal test graphs: {len(test)}")

    return DatasetBundle(train_dataset=train, val_dataset=val, test_dataset=test, input_dim=train.input_dim, num_classes=2, temporal_test_datasets=temporal_test_datasets)


def _resolve_bcg_root(args):
    root = os.path.abspath(os.path.expanduser(args.root))
    bcg_root = root if os.path.basename(os.path.normpath(root)).lower() == "bcg" else os.path.join(root, "BCG")

    if not os.path.isdir(bcg_root):
        raise FileNotFoundError(f'BCG directory was not found: {bcg_root}')

    required_files = ["BCG_hashed_FCGs.zip", "BCG_all_apk_features.json"]
    missing_files = [filename for filename in required_files if not os.path.isfile(os.path.join(bcg_root, filename))]
    if missing_files:
        raise FileNotFoundError(f'BCG directory is missing required files: {missing_files}. BCG root: {bcg_root}')

    return bcg_root


def _load_bcg_bundle(args):
    root = _resolve_bcg_root(args)
    mode = str(getattr(args, "bcg_split_mode", "cross_type")).lower()

    if mode == "intra_type":
        builder = build_bcg_intra_datasets
        setting = "intra_type_family"
    elif mode == "cross_type":
        builder = build_bcg_inter_datasets
        setting = "cross_type_family"
    else:
        raise ValueError(
            f"Unsupported BCG split mode: {mode}. Expected intra_type or cross_type."
        )

    train, val, test, split_info = builder(
        bcg_root=root,
        setting=setting,
        feature_mode=BCG_FEATURE_MODE,
        time_field=BCG_TIME_FIELD,
        seed=int(args.seed),
        train_ratio=BCG_TRAIN_RATIO,
        val_ratio=BCG_VAL_RATIO,
        test_ratio=BCG_TEST_RATIO,
        min_family_samples=BCG_MIN_FAMILY_SAMPLES,
        min_families_per_type=BCG_MIN_FAMILIES_PER_TYPE,
        min_type_samples=BCG_MIN_TYPE_SAMPLES,
        split_manifest_dir=getattr(args, "bcg_split_manifest_dir", None),
    )

    print(f"BCG dataset root: {root}")
    print(f"BCG family-OOD mode: {setting}")
    if split_info.get("manifest_path"):
        print(f"BCG split manifest: {split_info['manifest_path']}")

    return DatasetBundle(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        input_dim=int(train.input_dim),
        num_classes=2,
    )


def _format_bias_tag(value):
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_malnet_split_dir(args):
    explicit = getattr(args, "malnet_split_dir", None)
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))

    statistic = MALNET_SPLIT_STATISTIC
    direction = MALNET_SPLIT_DIRECTION
    val_mode = MALNET_VAL_MODE
    split_seed = MALNET_SPLIT_SEED
    bias_tag = _format_bias_tag(getattr(args, "bias", 0.33))
    folder_name = f"{statistic}_bias_{bias_tag}_seed_{split_seed}_{direction}_{val_mode}_val"
    malnet_data_dir = os.path.abspath(os.path.expanduser(args.malnet_data_dir))
    return os.path.join(malnet_data_dir, "MalNetTiny_OOD", folder_name)


def _validate_malnet_indices(indices, split_name, dataset_size):
    indices = torch.as_tensor(indices, dtype=torch.long).view(-1)

    if indices.numel() == 0:
        raise RuntimeError(f'MalNet-Tiny {split_name} split is empty.')
    if torch.unique(indices).numel() != indices.numel():
        raise RuntimeError(f'MalNet-Tiny {split_name} split contains duplicate indices.')

    minimum, maximum = int(indices.min().item()), int(indices.max().item())
    if minimum < 0 or maximum >= int(dataset_size):
        raise RuntimeError(f'MalNet-Tiny {split_name} indices must lie in [0, {int(dataset_size) - 1}], observed [{minimum}, {maximum}].')

    return indices


def _load_malnet_tiny_bundle(args):
    split_dir = _resolve_malnet_split_dir(args)
    split_path = os.path.join(split_dir, "split_indices.pt")

    if not os.path.isfile(split_path):
        raise FileNotFoundError(f'MalNet-Tiny split file was not found:\n  {split_path}\nPass --malnet-split-dir explicitly, or generate the split with the released num_nodes/small_to_large/source/seed=42 settings and matching --bias.')

    payload = _safe_torch_load(split_path)
    if not isinstance(payload, dict):
        raise TypeError(f'Expected a dictionary in {split_path}, got {type(payload).__name__}.')

    required_keys = {"train", "val", "test"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise KeyError(f'MalNet-Tiny split file is missing keys: {sorted(missing_keys)}.')

    stored_dataset_root = payload.get("dataset_root")
    local_dataset_root = os.path.join(args.malnet_data_dir, "MalNetTiny")

    if stored_dataset_root and os.path.isdir(str(stored_dataset_root)):
        dataset_root = os.path.abspath(str(stored_dataset_root))
    elif os.path.isdir(local_dataset_root):
        dataset_root = os.path.abspath(local_dataset_root)
    else:
        dataset_root = os.path.abspath(local_dataset_root)

    dataset = MalNetTiny(root=dataset_root, split=None)

    train_indices = _validate_malnet_indices(payload["train"], "train", len(dataset))
    val_indices = _validate_malnet_indices(payload["val"], "validation", len(dataset))
    test_indices = _validate_malnet_indices(payload["test"], "test", len(dataset))

    train_set, val_set, test_set = set(train_indices.tolist()), set(val_indices.tolist()), set(test_indices.tolist())
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError('MalNet-Tiny train/validation/test indices overlap.')

    train = dataset.index_select(train_indices.tolist())
    val = dataset.index_select(val_indices.tolist())
    test = dataset.index_select(test_indices.tolist())

    print(f"MalNet-Tiny dataset root: {dataset_root}")
    print(f"MalNet-Tiny OOD split: {split_path}")
    print(f"MalNet-Tiny split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    return DatasetBundle(train_dataset=train, val_dataset=val, test_dataset=test, input_dim=infer_input_dimension(train), num_classes=infer_num_classes(train))


def _extract_benign_label_from_metadata(label_names):
    if label_names is None:
        return None

    candidates = []

    if isinstance(label_names, dict):
        for key, value in label_names.items():
            if str(value).strip().lower() == "benign":
                try:
                    candidates.append(int(key))
                except (TypeError, ValueError):
                    pass

            if str(key).strip().lower() == "benign":
                try:
                    candidates.append(int(value))
                except (TypeError, ValueError):
                    pass
    elif isinstance(label_names, (list, tuple)):
        for label, name in enumerate(label_names):
            if str(name).strip().lower() == 'benign':
                candidates.append(int(label))

    candidates = sorted(set(candidates))
    if len(candidates) > 1:
        raise RuntimeError(f'Multiple MalNet-Tiny benign labels were found in label_names={label_names}: {candidates}')
    return candidates[0] if candidates else None


def _infer_malnet_label_names_from_raw(dataset):
    split_root = dataset.raw_paths[1]

    if not os.path.isdir(split_root):
        raise FileNotFoundError(f'MalNet-Tiny raw type split directory was not found: {split_root}')

    type_to_label = {}

    for split_name in ("train", "val", "test"):
        split_path = os.path.join(split_root, f"{split_name}.txt")
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f'MalNet-Tiny raw split file was not found: {split_path}')

        with open(split_path, "r", encoding="utf-8") as file:
            filenames = [line.strip() for line in file if line.strip()]

        for filename in filenames:
            normalized_filename = filename.replace("\\", "/")
            malware_type = normalized_filename.split("/")[0].strip()
            if not malware_type:
                raise RuntimeError(f'Could not extract malware type from MalNet-Tiny split entry: {filename!r}')
            if malware_type not in type_to_label:
                type_to_label[malware_type] = len(type_to_label)

    if not type_to_label:
        raise RuntimeError(f'No malware types were found under {split_root}.')

    return {int(label): str(name) for name, label in type_to_label.items()}


def _resolve_malnet_benign_label(payload, dataset):
    label_names = payload.get("label_names")
    benign_label = _extract_benign_label_from_metadata(label_names)

    if benign_label is not None:
        resolved_label_names = label_names
        source = "split metadata"
    else:
        resolved_label_names = _infer_malnet_label_names_from_raw(dataset)
        benign_label = _extract_benign_label_from_metadata(resolved_label_names)
        source = "PyG raw split files"

    if benign_label is None:
        raise RuntimeError(f'Could not identify the MalNet-Tiny benign label. Resolved label names: {resolved_label_names}')

    observed_labels = sorted({int(torch.as_tensor(dataset[index].y).view(-1)[0].item()) for index in range(len(dataset))})
    if benign_label not in observed_labels:
        raise RuntimeError(f'Resolved MalNet-Tiny benign label {benign_label} is absent from dataset labels {observed_labels}.')

    return int(benign_label), resolved_label_names, source


def _binary_malnet_distribution(dataset):
    labels = [int(torch.as_tensor(dataset[index].y).view(-1)[0].item()) for index in range(len(dataset))]
    values, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def _load_binary_malnet_bundle(args):
    split_dir = _resolve_malnet_split_dir(args)
    split_path = os.path.join(split_dir, "split_indices.pt")

    if not os.path.isfile(split_path):
        raise FileNotFoundError(f'Binary MalNet-Tiny split file was not found:\n  {split_path}\nThe binary task reuses the existing MalNet-Tiny OOD split. Pass --malnet-split-dir explicitly, or generate the split with the released num_nodes/small_to_large/source/seed=42 settings and matching --bias.')

    payload = _safe_torch_load(split_path)
    if not isinstance(payload, dict):
        raise TypeError(f'Expected a dictionary in {split_path}, got {type(payload).__name__}.')

    required_keys = {"train", "val", "test"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise KeyError(f'Binary MalNet-Tiny split file is missing keys: {sorted(missing_keys)}.')

    stored_dataset_root = payload.get("dataset_root")
    local_dataset_root = os.path.join(args.malnet_data_dir, "MalNetTiny")

    if stored_dataset_root and os.path.isdir(str(stored_dataset_root)):
        dataset_root = os.path.abspath(str(stored_dataset_root))
    elif os.path.isdir(local_dataset_root):
        dataset_root = os.path.abspath(local_dataset_root)
    else:
        dataset_root = os.path.abspath(local_dataset_root)

    dataset = MalNetTiny(root=dataset_root, split=None)

    train_indices = _validate_malnet_indices(payload["train"], "train", len(dataset))
    val_indices = _validate_malnet_indices(payload["val"], "validation", len(dataset))
    test_indices = _validate_malnet_indices(payload["test"], "test", len(dataset))

    train_set, val_set, test_set = set(train_indices.tolist()), set(val_indices.tolist()), set(test_indices.tolist())
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError('Binary MalNet-Tiny train/validation/test indices overlap.')

    benign_label, original_label_names, label_source = _resolve_malnet_benign_label(payload, dataset)

    train_original = dataset.index_select(train_indices.tolist())
    val_original = dataset.index_select(val_indices.tolist())
    test_original = dataset.index_select(test_indices.tolist())

    train = BinaryMalNetDataset(train_original, benign_label)
    val = BinaryMalNetDataset(val_original, benign_label)
    test = BinaryMalNetDataset(test_original, benign_label)

    train_distribution = _binary_malnet_distribution(train)
    val_distribution = _binary_malnet_distribution(val)
    test_distribution = _binary_malnet_distribution(test)

    if set(train_distribution) != {0, 1}:
        raise RuntimeError(f'Binary MalNet-Tiny training split must contain Benign=0 and Malware=1, observed {train_distribution}.')
    if not set(val_distribution).issubset({0, 1}):
        raise RuntimeError(f'Binary MalNet-Tiny validation labels are invalid: {val_distribution}.')
    if not set(test_distribution).issubset({0, 1}):
        raise RuntimeError(f'Binary MalNet-Tiny test labels are invalid: {test_distribution}.')

    print(f"Binary MalNet-Tiny dataset root: {dataset_root}")
    print(f"Binary MalNet-Tiny OOD split: {split_path}")
    print(f"Binary MalNet-Tiny bias: {getattr(args, 'bias', 0.33)}")
    print(f"Binary MalNet-Tiny original label names: {original_label_names}")
    print(f"Binary MalNet-Tiny benign original label: {benign_label}")
    print(f"Binary MalNet-Tiny benign-label source: {label_source}")
    print("Binary MalNet-Tiny task: Benign=0, Malware=1")
    print(f"Binary MalNet-Tiny train distribution: {train_distribution} | graphs={len(train)}")
    print(f"Binary MalNet-Tiny validation distribution: {val_distribution} | graphs={len(val)}")
    print(f"Binary MalNet-Tiny test distribution: {test_distribution} | graphs={len(test)}")

    return DatasetBundle(train_dataset=train, val_dataset=val, test_dataset=test, input_dim=infer_input_dimension(train), num_classes=2)


def _resolve_callgraph_root(args):
    explicit = getattr(args, "callgraph_data_dir", None)

    if explicit:
        root = os.path.abspath(os.path.expanduser(str(explicit)))
        if os.path.isfile(os.path.join(root, 'processed', 'data.pt')):
            return root

        candidate = os.path.join(root, "dataset-call-graph-blogpost-material-master")
        if os.path.isfile(os.path.join(candidate, 'processed', 'data.pt')):
            return candidate

        raise FileNotFoundError(f"CallGraph processed/data.pt was not found under --callgraph-data-dir: {root}")

    root = os.path.abspath(os.path.expanduser(args.root))

    if os.path.isfile(os.path.join(root, 'processed', 'data.pt')):
        return root

    candidate = os.path.join(root, "dataset-call-graph-blogpost-material-master")
    if os.path.isfile(os.path.join(candidate, 'processed', 'data.pt')):
        return candidate

    raise FileNotFoundError(f"CallGraph dataset was not found. Expected either:\n  {os.path.join(root, 'processed', 'data.pt')}\nor:\n  {os.path.join(candidate, 'processed', 'data.pt')}")


def _resolve_callgraph_split_dir(args, dataset_root):
    explicit = getattr(args, "callgraph_split_dir", None)
    if explicit:
        return os.path.abspath(os.path.expanduser(str(explicit)))

    statistic = CALLGRAPH_SPLIT_STATISTIC
    direction = CALLGRAPH_SPLIT_DIRECTION
    val_mode = CALLGRAPH_VAL_MODE
    split_seed = CALLGRAPH_SPLIT_SEED
    bias_tag = _format_bias_tag(getattr(args, "bias", 0.33))
    folder_name = f"{statistic}_bias_{bias_tag}_seed_{split_seed}_{direction}_{val_mode}_val"

    explicit_ood_root = getattr(args, "callgraph_ood_dir", None)

    if explicit_ood_root:
        ood_root = os.path.abspath(os.path.expanduser(str(explicit_ood_root)))
    else:
        ood_root = os.path.join(os.path.dirname(dataset_root), "CallGraph_OOD")

    return os.path.join(ood_root, folder_name)


def _load_callgraph_graphs(dataset_root):
    processed_path = os.path.join(dataset_root, "processed", "data.pt")
    payload = _safe_torch_load(processed_path)

    if not isinstance(payload, tuple) or len(payload) != 2:
        raise TypeError(f'Expected CallGraph {processed_path} to contain a (data, slices) tuple, got {type(payload).__name__}.')

    data, slices = payload
    required_keys = {"x", "edge_index", "y"}

    if not required_keys.issubset(set(data.keys())):
        raise RuntimeError(f'CallGraph data.pt must contain {sorted(required_keys)}, observed {sorted(data.keys())}.')
    if not required_keys.issubset(set(slices.keys())):
        raise RuntimeError(f'CallGraph data.pt slices must contain {sorted(required_keys)}, observed {sorted(slices.keys())}.')

    num_graphs = int(slices["y"].numel() - 1)
    if num_graphs != 1361:
        raise RuntimeError(f'Expected 1361 CallGraph graphs, observed {num_graphs}.')

    graphs = []

    for index in range(num_graphs):
        graph = separate(cls=data.__class__, batch=data, idx=index, slice_dict=slices, decrement=False)

        x = getattr(graph, "x", None)
        edge_index = getattr(graph, "edge_index", None)
        target = getattr(graph, "y", None)

        if x is None:
            raise RuntimeError(f'CallGraph graph {index} does not contain x.')
        if x.dim() != 2 or int(x.size(1)) != 27:
            raise RuntimeError(f'CallGraph graph {index} must have node features [N, 27], got {tuple(x.shape)}.')
        if edge_index is None or edge_index.dim() != 2 or int(edge_index.size(0)) != 2:
            raise RuntimeError(f'CallGraph graph {index} has invalid edge_index.')
        if target is None:
            raise RuntimeError(f'CallGraph graph {index} does not contain y.')

        graph.x = x.float()
        graph.edge_index = edge_index.long()
        graph.y = torch.as_tensor(target).view(-1).long()

        if graph.y.numel() != 1:
            raise RuntimeError(f'CallGraph graph {index} must contain exactly one label, got {graph.y.numel()}.')

        num_nodes = int(graph.x.size(0))
        graph.num_nodes = num_nodes
        graph.graph_id = torch.tensor([index], dtype=torch.long)

        if graph.edge_index.numel() > 0:
            minimum, maximum = int(graph.edge_index.min().item()), int(graph.edge_index.max().item())

            if minimum < 0 or maximum >= num_nodes:
                node_start = int(slices["x"][index])
                node_end = int(slices["x"][index + 1])

                if minimum >= node_start and maximum < node_end:
                    graph.edge_index = graph.edge_index - node_start
                    minimum, maximum = int(graph.edge_index.min().item()), int(graph.edge_index.max().item())

                if minimum < 0 or maximum >= num_nodes:
                    raise RuntimeError(f'CallGraph graph {index} has edge indices outside [0, {num_nodes - 1}]: observed [{minimum}, {maximum}].')

        graphs.append(graph)

    labels = torch.tensor([int(graph.y.item()) for graph in graphs], dtype=torch.long)
    classes, counts = torch.unique(labels, sorted=True, return_counts=True)

    if classes.tolist() != [0, 1]:
        raise RuntimeError(f'CallGraph labels must be [0, 1], observed {classes.tolist()}.')
    if sorted(counts.tolist()) != [546, 815]:
        raise RuntimeError(f'Expected CallGraph class sizes [546, 815], observed {dict(zip(classes.tolist(), counts.tolist()))}.')

    return graphs


def _validate_callgraph_indices(indices, split_name, dataset_size):
    indices = torch.as_tensor(indices, dtype=torch.long).view(-1)

    if indices.numel() == 0:
        raise RuntimeError(f'CallGraph {split_name} split is empty.')
    if torch.unique(indices).numel() != indices.numel():
        raise RuntimeError(f'CallGraph {split_name} split contains duplicate indices.')

    minimum, maximum = int(indices.min().item()), int(indices.max().item())
    if minimum < 0 or maximum >= int(dataset_size):
        raise RuntimeError(f'CallGraph {split_name} indices must lie in [0, {int(dataset_size) - 1}], observed [{minimum}, {maximum}].')

    return indices


def _validate_callgraph_split(graphs, train_indices, val_indices, test_indices):
    train_set, val_set, test_set = set(train_indices.tolist()), set(val_indices.tolist()), set(test_indices.tolist())

    if train_set & val_set:
        raise RuntimeError('CallGraph train and validation indices overlap.')
    if train_set & test_set:
        raise RuntimeError('CallGraph train and test indices overlap.')
    if val_set & test_set:
        raise RuntimeError('CallGraph validation and test indices overlap.')

    union = train_set | val_set | test_set
    if len(union) != len(graphs):
        raise RuntimeError(f'CallGraph split covers {len(union)} graphs, but dataset contains {len(graphs)}.')

    labels = np.asarray([int(graph.y.item()) for graph in graphs], dtype=np.int64)
    unique_labels, full_counts = np.unique(labels, return_counts=True)
    class_sizes = {int(label): int(count) for label, count in zip(unique_labels, full_counts)}

    expected_by_class_size = {
        546: {"train": 382, "val": 55, "test": 109},
        815: {"train": 571, "val": 81, "test": 163},
    }

    for split_name, indices in {"train": train_indices, "val": val_indices, "test": test_indices}.items():
        split_labels = labels[indices.numpy()]
        split_unique, split_counts = np.unique(split_labels, return_counts=True)
        actual = {int(label): int(count) for label, count in zip(split_unique, split_counts)}

        for label in unique_labels:
            label = int(label)
            class_size = class_sizes[label]

            if class_size not in expected_by_class_size:
                raise RuntimeError(f'Unexpected CallGraph class size {class_size} for label {label}.')

            expected = expected_by_class_size[class_size][split_name]
            observed = actual.get(label, 0)

            if observed != expected:
                raise RuntimeError(f'CallGraph {split_name} label {label} contains {observed} graphs; expected {expected}.')


def _load_callgraph_bundle(args):
    dataset_root = _resolve_callgraph_root(args)
    split_dir = _resolve_callgraph_split_dir(args, dataset_root)
    split_path = os.path.join(split_dir, "split_indices.pt")

    if not os.path.isfile(split_path):
        raise FileNotFoundError(
            f"CallGraph OOD split file was not found:\n  {split_path}\n"
            f"Current bias={getattr(args, 'bias', 0.33)}.\n"
            "Generate the split first or pass --callgraph-split-dir explicitly."
        )

    payload = _safe_torch_load(split_path)
    if not isinstance(payload, dict):
        raise TypeError(f'Expected a dictionary in {split_path}, got {type(payload).__name__}.')

    required_keys = {"train", "val", "test"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise KeyError(f'CallGraph split file is missing keys: {sorted(missing_keys)}.')

    graphs = _load_callgraph_graphs(dataset_root)
    train_indices = _validate_callgraph_indices(payload["train"], "train", len(graphs))
    val_indices = _validate_callgraph_indices(payload["val"], "validation", len(graphs))
    test_indices = _validate_callgraph_indices(payload["test"], "test", len(graphs))

    _validate_callgraph_split(graphs, train_indices, val_indices, test_indices)

    train = CallGraphDataset(graphs, train_indices.tolist())
    val = CallGraphDataset(graphs, val_indices.tolist())
    test = CallGraphDataset(graphs, test_indices.tolist())

    labels = np.asarray([int(graph.y.item()) for graph in graphs], dtype=np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_names = {}

    for label, count in zip(unique_labels, counts):
        if int(count) == 546: label_names[int(label)] = "goodware"
        elif int(count) == 815: label_names[int(label)] = "malware"
        else: label_names[int(label)] = f"class_{int(label)}"

    print(f"CallGraph dataset root: {dataset_root}")
    print(f"CallGraph processed graph file: {os.path.join(dataset_root, 'processed', 'data.pt')}")
    print(f"CallGraph OOD split: {split_path}")
    print(f"CallGraph bias: {getattr(args, 'bias', 0.33)}")
    print(f"CallGraph node feature dimension: 27")
    print(f"CallGraph labels: {label_names}")
    print(f"CallGraph split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    for split_name, split_dataset in (("train", train), ("val", val), ("test", test)):
        split_labels = [int(split_dataset[index].y.item()) for index in range(len(split_dataset))]
        values, split_counts = np.unique(split_labels, return_counts=True)
        distribution = {int(label): int(count) for label, count in zip(values, split_counts)}
        nodes = np.asarray([int(split_dataset[index].num_nodes) for index in range(len(split_dataset))], dtype=np.float64)
        print(f"CallGraph {split_name}: labels={distribution} | nodes mean={nodes.mean():.2f} | median={np.median(nodes):.2f}")

    return DatasetBundle(train_dataset=train, val_dataset=val, test_dataset=test, input_dim=27, num_classes=2)


def _validate_dataset_bundle(bundle):
    if len(bundle.train_dataset) == 0:
        raise RuntimeError('The source training dataset is empty.')
    if len(bundle.val_dataset) == 0:
        raise RuntimeError('The validation dataset is empty.')
    if len(bundle.test_dataset) == 0:
        raise RuntimeError('The OOD test dataset is empty.')
    if int(bundle.input_dim) < 1:
        raise RuntimeError(f'Dataset input_dim must be positive, got {bundle.input_dim}.')
    if int(bundle.num_classes) < 2:
        raise RuntimeError(f'Dataset num_classes must be at least 2, got {bundle.num_classes}.')

    observed_input_dim = infer_input_dimension(bundle.train_dataset)
    if observed_input_dim != int(bundle.input_dim):
        raise RuntimeError(f'Declared input_dim={bundle.input_dim}, but source training graphs have input dimension {observed_input_dim}.')

    if len(bundle.train_dataset) <= 10000:
        observed_num_classes = infer_num_classes(bundle.train_dataset)
        if observed_num_classes != int(bundle.num_classes):
            raise RuntimeError(f'Declared num_classes={bundle.num_classes}, but source training contains {observed_num_classes} classes.')
    else:
        print(f"Skipping full class scan for large dataset ({len(bundle.train_dataset)} graphs).")

    validation_limit = 100 if len(bundle.train_dataset) > 10000 else None
    if validation_limit is not None:
        print(f'Large dataset detected: validating at most {validation_limit} graphs per split.')

    _validate_graph_collection("Train", bundle.train_dataset, bundle.input_dim, max_graphs=validation_limit)
    _validate_graph_collection("Validation", bundle.val_dataset, bundle.input_dim, max_graphs=validation_limit)

    if bundle.temporal_test_datasets is None:
        _validate_graph_collection("OOD test", bundle.test_dataset, bundle.input_dim, max_graphs=validation_limit)
    else:
        if not isinstance(bundle.temporal_test_datasets, dict):
            raise TypeError('temporal_test_datasets must be a dictionary mapping test environment names to datasets.')
        if not bundle.temporal_test_datasets:
            raise RuntimeError('temporal_test_datasets must contain at least one temporal OOD test dataset.')

        for test_name, dataset in bundle.temporal_test_datasets.items():
            if len(dataset) == 0:
                raise RuntimeError(f"Temporal OOD test dataset '{test_name}' is empty.")
            _validate_graph_collection(f"OOD test {test_name}", dataset, bundle.input_dim, max_graphs=validation_limit)


def _validate_graph_collection(name, dataset, input_dim, max_graphs=None):
    num_graphs = len(dataset) if max_graphs is None else min(len(dataset), int(max_graphs))

    for index in range(num_graphs):
        data = dataset[index]
        if not isinstance(data, Data):
            raise TypeError(f'{name} item {index} must be a PyG Data object, got {type(data).__name__}.')

        num_nodes = int(data.num_nodes)
        if num_nodes < 1:
            raise RuntimeError(f'{name} graph {index} contains no nodes.')

        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise RuntimeError(f'{name} graph {index} has no edge_index.')
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise RuntimeError(f'{name} graph {index} edge_index must have shape [2, E], got {edge_index.shape}.')

        if edge_index.numel() > 0:
            minimum, maximum = int(edge_index.min().item()), int(edge_index.max().item())
            if minimum < 0 or maximum >= num_nodes:
                raise RuntimeError(f'{name} graph {index} has edge indices outside [0, {num_nodes - 1}]: observed [{minimum}, {maximum}].')

        x = getattr(data, "x", None)

        if x is None:
            if int(input_dim) != 1:
                raise RuntimeError(f'{name} graph {index} has no node features, but the declared input dimension is {input_dim}.')
        else:
            if x.dim() == 1:
                feature_dim = 1
            elif x.dim() == 2:
                feature_dim = int(x.size(1))
            else:
                raise RuntimeError(f"{name} graph {index} node features must have shape [N, F], got {x.shape}.")

            if feature_dim != int(input_dim):
                raise RuntimeError(f'{name} graph {index} has feature dimension {feature_dim}, expected {input_dim}.')

        target = getattr(data, "y", None)
        if target is None:
            raise RuntimeError(f'{name} graph {index} does not contain y.')

        target = torch.as_tensor(target).view(-1)
        valid_target = target[torch.isfinite(target)]
        if valid_target.numel() != 1:
            raise RuntimeError(f'{name} graph {index} must contain exactly one finite classification label, got {valid_target.numel()}.')


def collect_dataset_labels(dataset):
    labels = []

    for index in range(len(dataset)):
        target = getattr(dataset[index], "y", None)
        if target is None:
            raise RuntimeError(f'Graph {index} does not contain y.')

        target = torch.as_tensor(target).detach().view(-1)
        valid_target = target[torch.isfinite(target)]
        if valid_target.numel() > 0:
            labels.append(valid_target.long().cpu())

    if not labels:
        raise RuntimeError('The dataset contains no valid labels.')
    return torch.cat(labels)


def print_label_statistics(name, dataset):
    labels = collect_dataset_labels(dataset)
    values, counts = torch.unique(labels, sorted=True, return_counts=True)
    distribution = {int(value): int(count) for value, count in zip(values, counts)}

    print(f"{name} labels: {values.tolist()}")
    print(f"{name} label distribution: {distribution}")
    return values


def validate_dataset_classes(train_classes, val_classes, test_classes):
    train_classes = torch.as_tensor(train_classes).long().cpu().view(-1)
    val_classes = torch.as_tensor(val_classes).long().cpu().view(-1)
    test_classes = torch.as_tensor(test_classes).long().cpu().view(-1)

    train_class_values = train_classes.tolist()
    train_class_set = set(train_class_values)

    unseen_val_classes = sorted(set(val_classes.tolist()) - train_class_set)
    unseen_test_classes = sorted(set(test_classes.tolist()) - train_class_set)

    if unseen_val_classes:
        raise RuntimeError(f'Validation set contains classes absent from training: {unseen_val_classes}.')
    if unseen_test_classes:
        raise RuntimeError(f'OOD test set contains classes absent from training: {unseen_test_classes}.')

    expected_classes = list(range(len(train_class_values)))
    if train_class_values != expected_classes:
        raise RuntimeError(f'Class labels must be contiguous and start at zero for cross-entropy. Observed classes={train_class_values}, expected={expected_classes}.')


def shutdown_loader_workers(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return

    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()

    loader._iterator = None
