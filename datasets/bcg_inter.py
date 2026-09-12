import itertools
import json
import os
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import LocalDegreeProfile


BCG_SPLIT_IMPLEMENTATION = "family_ood_type_first_v4_fastcache"
_LABEL_TO_INDEX = {"benign": 0, "malware": 1}
_SHARED_BCG_DATASETS = {}
_CACHE_VERSION = "v1"


class _BCGAllDataset(InMemoryDataset):
    def __init__(self, bcg_root, feature_mode="ldp", time_field="vt_year"):
        self.bcg_root = os.path.abspath(os.path.expanduser(bcg_root))
        self.feature_mode = str(feature_mode).lower()
        self.time_field = str(time_field)
        if self.feature_mode not in {'ldp', 'constant'}:
            raise ValueError(f"Unsupported BCG feature mode: {self.feature_mode}. Expected 'ldp' or 'constant'.")
        if self.time_field not in {'vt_year', 'apk_year_only', 'dex_year'}:
            raise ValueError(f'Unsupported BCG time field: {self.time_field}.')
        self.input_dim = 5 if self.feature_mode == "ldp" else 1
        self.declared_num_classes = 2
        self.large_scale = False
        self.graph_zip_path = os.path.join(self.bcg_root, "BCG_hashed_FCGs.zip")
        self.feature_json_path = os.path.join(self.bcg_root, "BCG_all_apk_features.json")
        if not os.path.isfile(self.graph_zip_path):
            raise FileNotFoundError(f'BCG hashed FCG archive was not found: {self.graph_zip_path}')
        if not os.path.isfile(self.feature_json_path):
            raise FileNotFoundError(f'BCG APK feature JSONL file was not found: {self.feature_json_path}')
        processed_root = os.path.join(self.bcg_root, "iec_good_cache", f"bcg_{self.time_field}_{self.feature_mode}_{_CACHE_VERSION}")
        super().__init__(root=processed_root)
        self.load(self.processed_paths[0])
        self._load_processed_metadata()
        self._upgrade_family_metadata_if_needed()
        if len(self.graph_metadata) != len(self):
            raise RuntimeError(f'BCG metadata/data size mismatch: metadata={len(self.graph_metadata)}, graphs={len(self)}.')
        self._year_to_indices = None

    @property
    @property
    def raw_file_names(self):
        return []

    @property
    @property
    def processed_file_names(self):
        return ['data.pt', 'metadata.json']

    def process(self):
        print("=" * 80)
        print("Processing BCG dataset")
        print(f"BCG root: {self.bcg_root}")
        print(f"Graph archive: {self.graph_zip_path}")
        print(f"APK feature file: {self.feature_json_path}")
        print(f"Feature mode: {self.feature_mode}")
        print(f"Temporal field: {self.time_field}")
        print("=" * 80)
        records = self._load_json_records()
        data_list, graph_metadata = [], []
        json_shas = set(records)
        with zipfile.ZipFile(self.graph_zip_path, "r") as archive:
            graph_members = self._collect_graph_members(archive)
            graph_shas = set(graph_members)
            missing_graphs, missing_json = json_shas - graph_shas, graph_shas - json_shas
            print(f"JSON records: {len(json_shas)}")
            print(f"FCG files: {len(graph_shas)}")
            print(f"JSON without graph: {len(missing_graphs)}")
            print(f"Graph without JSON: {len(missing_json)}")
            if missing_graphs:
                raise RuntimeError(f'BCG JSON records without matching FCG files. Sample: {sorted(missing_graphs)[:10]}')
            if missing_json:
                raise RuntimeError(f'BCG FCG files without matching JSON records. Sample: {sorted(missing_json)[:10]}')
            sorted_shas = sorted(graph_shas)
            ldp_transform = LocalDegreeProfile() if self.feature_mode == "ldp" else None
            json_edge_mismatch_count = header_edge_mismatch_count = node_count_mismatch_count = fallback_indexing_count = 0
            for graph_id, sha256 in enumerate(sorted_shas):
                record = records[sha256]
                member_name = graph_members[sha256]
                label = self._parse_binary_label(record, sha256)
                year = self._parse_year(record, sha256)
                malware_type, family = self._parse_malware_group_labels(record, label)
                json_num_nodes, json_num_edges = self._safe_int(record.get("nodes")), self._safe_int(record.get("edges"))
                edge_index, num_nodes, header_num_nodes, header_num_edges, indexing_mode = self._parse_edgelist(archive.read(member_name), member_name, json_num_nodes)
                if indexing_mode == 'fallback_remap':
                    fallback_indexing_count += 1
                parsed_num_edges = int(edge_index.size(1))
                if json_num_edges is not None and json_num_edges != parsed_num_edges:
                    json_edge_mismatch_count += 1
                if header_num_edges is not None and header_num_edges != parsed_num_edges:
                    header_edge_mismatch_count += 1
                if json_num_nodes is not None and header_num_nodes is not None and (json_num_nodes != header_num_nodes):
                    node_count_mismatch_count += 1
                data = Data(edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), year=torch.tensor([year], dtype=torch.long), graph_id=torch.tensor([graph_id], dtype=torch.long), num_nodes=num_nodes)
                if self.feature_mode == "ldp":
                    data = ldp_transform(data)
                    data.x = torch.nan_to_num(data.x.float(), nan=0.0, posinf=0.0, neginf=0.0)
                else: data.x = torch.ones((num_nodes, 1), dtype=torch.float32)
                if data.x.dim() != 2 or data.x.size(0) != num_nodes or data.x.size(1) != self.input_dim:
                    raise RuntimeError(f'Invalid BCG node feature shape for {sha256}: expected [{num_nodes}, {self.input_dim}], got {tuple(data.x.shape)}.')
                if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
                    raise RuntimeError(f'Invalid BCG edge_index shape for {sha256}: {tuple(data.edge_index.shape)}.')
                if data.edge_index.numel() > 0:
                    minimum, maximum = int(data.edge_index.min().item()), int(data.edge_index.max().item())
                    if minimum < 0 or maximum >= num_nodes:
                        raise RuntimeError(f'BCG edge index out of range for {sha256}: node range=[0, {num_nodes - 1}], observed=[{minimum}, {maximum}].')
                data_list.append(data)
                graph_metadata.append({"graph_id": graph_id, "sha256": sha256, "year": year, "label": label, "binary_type": "benign" if label == 0 else "malware", "malware_type": malware_type, "family": family, "num_nodes": num_nodes, "num_edges": parsed_num_edges, "json_num_edges": json_num_edges, "header_num_edges": header_num_edges, "indexing_mode": indexing_mode})
                if (graph_id + 1) % 500 == 0 or graph_id + 1 == len(sorted_shas):
                    print(f'Processed BCG graphs: {graph_id + 1}/{len(sorted_shas)}')
        if not data_list:
            raise RuntimeError('BCG processing produced no valid graphs.')
        self.save(data_list, self.processed_paths[0])
        metadata = {"cache_version": _CACHE_VERSION, "feature_mode": self.feature_mode, "time_field": self.time_field, "input_dim": self.input_dim, "num_classes": 2, "num_graphs": len(data_list), "graphs": graph_metadata}
        with open(self.processed_paths[1], 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)
        print("=" * 80)
        print("BCG processing completed")
        print(f"Graphs: {len(data_list)}")
        print(f"Input dimension: {self.input_dim}")
        print(f"JSON edge-count mismatches: {json_edge_mismatch_count}")
        print(f"Header edge-count mismatches: {header_edge_mismatch_count}")
        print(f"JSON/header node-count mismatches: {node_count_mismatch_count}")
        print(f"Fallback node-ID remappings: {fallback_indexing_count}")
        print(f"Processed graph cache: {self.processed_paths[0]}")
        print(f"Processed metadata: {self.processed_paths[1]}")
        print("=" * 80)

    def _load_processed_metadata(self):
        metadata_path = self.processed_paths[1]
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f'BCG processed metadata was not found: {metadata_path}')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        stored_feature_mode, stored_time_field, stored_input_dim = str(metadata.get("feature_mode")), str(metadata.get("time_field")), int(metadata.get("input_dim"))
        if stored_feature_mode != self.feature_mode:
            raise RuntimeError(f'BCG cache feature mode mismatch: expected {self.feature_mode}, found {stored_feature_mode}.')
        if stored_time_field != self.time_field:
            raise RuntimeError(f'BCG cache time field mismatch: expected {self.time_field}, found {stored_time_field}.')
        if stored_input_dim != self.input_dim:
            raise RuntimeError(f'BCG cache input dimension mismatch: expected {self.input_dim}, found {stored_input_dim}.')
        self.graph_metadata = metadata.get("graphs", [])

    def _upgrade_family_metadata_if_needed(self):
        if not self.graph_metadata:
            return
        if not any(('malware_type' not in item or 'family' not in item for item in self.graph_metadata)):
            return
        print("Upgrading existing BCG metadata with malware type/family labels (graph tensors are not recomputed).")
        records = self._load_json_records()
        for item in self.graph_metadata:
            sha256 = str(item["sha256"]).lower()
            record = records.get(sha256)
            if record is None:
                raise RuntimeError(f'Cannot upgrade BCG metadata: missing JSON record for {sha256}.')
            malware_type, family = self._parse_malware_group_labels(record, int(item["label"]))
            item["malware_type"], item["family"] = malware_type, family
        metadata = {"cache_version": _CACHE_VERSION, "feature_mode": self.feature_mode, "time_field": self.time_field, "input_dim": self.input_dim, "num_classes": 2, "num_graphs": len(self.graph_metadata), "graphs": self.graph_metadata}
        with open(self.processed_paths[1], 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)
        print(f"BCG metadata upgraded: {self.processed_paths[1]}")

    def indices_for_years(self, years):
        requested_years = {int(year) for year in years}
        if not requested_years:
            raise ValueError('BCG years must contain at least one year.')
        if self._year_to_indices is None:
            self._year_to_indices = {}
            for index, metadata in enumerate(self.graph_metadata):
                self._year_to_indices.setdefault(int(metadata['year']), []).append(index)
        available_years = set(self._year_to_indices)
        missing_years = requested_years - available_years
        if missing_years:
            raise ValueError(f'Requested BCG years are unavailable: {sorted(missing_years)}. Available years: {sorted(available_years)}.')
        indices = []
        for year in sorted(requested_years):
            indices.extend(self._year_to_indices[year])
        return indices

    def metadata_at(self, index):
        return self.graph_metadata[int(index)]

    def _load_json_records(self):
        records = {}
        with open(self.feature_json_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try: record = json.loads(line)
                except json.JSONDecodeError as exc: raise RuntimeError(f"Invalid JSON in {self.feature_json_path} at line {line_number}: {exc}") from exc
                sha256 = str(record.get("sha256", "")).strip().lower()
                if not sha256:
                    raise RuntimeError(f'BCG JSON record at line {line_number} does not contain a valid sha256.')
                if sha256 in records:
                    raise RuntimeError(f'Duplicate BCG sha256 in JSONL: {sha256}')
                records[sha256] = record
        if not records:
            raise RuntimeError(f'No BCG records were found in {self.feature_json_path}.')
        return records

    @staticmethod
    def _collect_graph_members(archive):
        members = {}
        for member_name in archive.namelist():
            if not member_name.lower().endswith('.edgelist'):
                continue
            sha256 = Path(member_name).stem.lower()
            if sha256 in members:
                raise RuntimeError(f'Duplicate BCG graph sha256 in ZIP archive: {sha256}')
            members[sha256] = member_name
        if not members:
            raise RuntimeError('No .edgelist files were found in BCG_hashed_FCGs.zip.')
        return members

    @staticmethod
    def _safe_int(value):
        if value is None:
            return None
        try: return int(value)
        except (TypeError, ValueError): return None

    @staticmethod
    def _normalize_group_label(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)): value = "++".join(sorted({str(item).strip().lower() for item in value if str(item).strip()}))
        else: value = str(value).strip().lower()
        if not value or value in {'none', 'null', 'nan', 'unknown', 'unclassified', 'undefined'}:
            return None
        return value

    @classmethod
    def _first_valid_group_label(cls, record, keys):
        for key in keys:
            if key not in record:
                continue
            value = cls._normalize_group_label(record.get(key))
            if value is not None:
                return value
        return None

    @classmethod
    def _parse_malware_group_labels(cls, record, binary_label):
        if int(binary_label) == 0:
            return (None, None)
        return cls._first_valid_group_label(record, ("final_label", "malware_type", "type_label")), cls._first_valid_group_label(record, ("family_label", "family", "malware_family"))

    @staticmethod
    def _parse_binary_label(record, sha256):
        binary_type = str(record.get("binary_type", "")).strip().lower()
        if binary_type not in _LABEL_TO_INDEX:
            raise ValueError(f'Invalid BCG binary_type for {sha256}: {binary_type!r}')
        return _LABEL_TO_INDEX[binary_type]

    def _parse_year(self, record, sha256):
        value = record.get(self.time_field)
        if value is None:
            raise ValueError(f'BCG record {sha256} does not contain {self.time_field}.')
        try: return int(value)
        except (TypeError, ValueError) as exc: raise ValueError(f"Invalid BCG {self.time_field} for {sha256}: {value!r}") from exc

    @staticmethod
    def _parse_edgelist(raw_content, member_name, json_num_nodes=None):
        try: text = raw_content.decode("utf-8")
        except UnicodeDecodeError: text = raw_content.decode("utf-8", errors="replace")
        edges, header_num_nodes, header_num_edges = [], None, None
        for line_number, raw_line in enumerate(text.splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                node_match = re.search(r"\bNodes:\s*(\d+)", line, flags=re.IGNORECASE)
                edge_match = re.search(r"\bEdges:\s*(\d+)", line, flags=re.IGNORECASE)
                if node_match:
                    header_num_nodes = int(node_match.group(1))
                if edge_match:
                    header_num_edges = int(edge_match.group(1))
                continue
            parts = line.split()
            if len(parts) < 2:
                raise RuntimeError(f'Malformed BCG edge in {member_name} at line {line_number}: {raw_line!r}')
            try: source, target = int(parts[0]), int(parts[1])
            except ValueError as exc: raise RuntimeError(f"Non-integer BCG edge in {member_name} at line {line_number}: {raw_line!r}") from exc
            edges.append((source, target))
        candidates = []
        if json_num_nodes is not None and json_num_nodes > 0:
            candidates.append(int(json_num_nodes))
        if header_num_nodes is not None and header_num_nodes > 0:
            candidates.append(int(header_num_nodes))
        if candidates: num_nodes = max(candidates)
        elif edges: num_nodes = len({node for edge in edges for node in edge})
        else: raise RuntimeError(f"Unable to determine the number of nodes for BCG graph: {member_name}")
        if num_nodes <= 0:
            raise RuntimeError(f'BCG graph contains no nodes: {member_name}')
        if not edges:
            return (torch.empty((2, 0), dtype=torch.long), num_nodes, header_num_nodes, header_num_edges, 'empty')
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        minimum, maximum = int(edge_index.min().item()), int(edge_index.max().item())
        if minimum >= 1 and maximum <= num_nodes: edge_index, indexing_mode = edge_index - 1, "one_based"
        elif minimum >= 0 and maximum < num_nodes: indexing_mode = "zero_based"
        else:
            unique_node_ids = sorted(set(edge_index.reshape(-1).tolist()))
            if len(unique_node_ids) > num_nodes:
                num_nodes = len(unique_node_ids)
            mapping = {node_id: index for index, node_id in enumerate(unique_node_ids)}
            edge_index = torch.tensor([mapping[int(node_id)] for node_id in edge_index.reshape(-1).tolist()], dtype=torch.long).view_as(edge_index).contiguous()
            indexing_mode = "fallback_remap"
        return edge_index.long().contiguous(), int(num_nodes), header_num_nodes, header_num_edges, indexing_mode


def _get_shared_bcg_dataset(bcg_root, feature_mode, time_field):
    normalized_root = os.path.abspath(os.path.expanduser(bcg_root))
    key = (normalized_root, str(feature_mode).lower(), str(time_field))
    if key not in _SHARED_BCG_DATASETS:
        _SHARED_BCG_DATASETS[key] = _BCGAllDataset(bcg_root=normalized_root, feature_mode=feature_mode, time_field=time_field)
    return _SHARED_BCG_DATASETS[key]


class BCGDataset(Dataset):
    def __init__(self, bcg_root, split_name, feature_mode="ldp", time_field="vt_year", years=None, indices=None):
        self.bcg_root = os.path.abspath(os.path.expanduser(bcg_root))
        self.split_name, self.feature_mode, self.time_field = str(split_name), str(feature_mode).lower(), str(time_field)
        if years is not None and indices is not None:
            raise ValueError('Specify either BCG years or indices, not both.')
        if years is None and indices is None:
            raise ValueError('BCGDataset requires either years or indices.')
        self._dataset = _get_shared_bcg_dataset(self.bcg_root, self.feature_mode, self.time_field)
        if indices is not None:
            self.years = None
            selected = [int(index) for index in indices]
            if len(set(selected)) != len(selected):
                raise ValueError(f'BCG split {self.split_name!r} contains duplicate indices.')
            if selected and (min(selected) < 0 or max(selected) >= len(self._dataset)):
                raise IndexError(f'BCG split {self.split_name!r} contains out-of-range indices.')
            self._indices = tuple(selected)
        else:
            self.years = tuple(sorted({int(year) for year in years}))
            if not self.years:
                raise ValueError('BCGDataset requires at least one year.')
            self._indices = tuple(self._dataset.indices_for_years(self.years))
        if not self._indices:
            raise RuntimeError(f'BCG split {self.split_name!r} is empty.')
        self.input_dim = self._dataset.input_dim
        self.declared_num_classes = self.num_classes = 2
        self.large_scale = False
        self._print_split_statistics()

    def _print_split_statistics(self):
        years, families, malware_types = {}, set(), set()
        benign = malware = 0
        for dataset_index in self._indices:
            metadata = self._dataset.metadata_at(dataset_index)
            year, label = int(metadata["year"]), int(metadata["label"])
            years.setdefault(year, {"benign": 0, "malware": 0})
            if label == 0:
                benign += 1
                years[year]["benign"] += 1
            elif label == 1:
                malware += 1
                years[year]["malware"] += 1
                if metadata.get('family'):
                    families.add(str(metadata['family']))
                if metadata.get('malware_type'):
                    malware_types.add(str(metadata['malware_type']))
            else: raise RuntimeError(f"Unexpected BCG label: {label}")
        total = benign + malware
        print(f"\n{'=' * 80}")
        print(f"BCG dataset statistics | split={self.split_name} | feature_mode={self.feature_mode}")
        print(f"{'=' * 80}")
        print(f"Total={total} | Benign={benign} ({100.0 * benign / total:.2f}%) | Malware={malware} ({100.0 * malware / total:.2f}%)")
        print(f"Malware families={len(families)} | Malware types={len(malware_types)}")
        print("Year composition: " + ", ".join(f"{year}:B{counts['benign']}/M{counts['malware']}" for year, counts in sorted(years.items())))
        print(f"{'=' * 80}\n")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f'BCGDataset index out of range: {index}')
        return self._dataset[self._indices[index]]

    def __getitems__(self, indices):
        return [self[index] for index in indices]

    def metadata_at(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f'BCGDataset index out of range: {index}')
        return self._dataset.metadata_at(self._indices[index])

    def sha256_at(self, index):
        return self.metadata_at(index)['sha256']

    @property
    @property
    def selected_indices(self):
        return tuple(self._indices)

    def _format_years(self):
        if self.years is None:
            return 'index-selected'
        if len(self.years) == 1:
            return str(self.years[0])
        if all((self.years[i] + 1 == self.years[i + 1] for i in range(len(self.years) - 1))):
            return f'{self.years[0]}-{self.years[-1]}'
        return ",".join(str(year) for year in self.years)

    def __repr__(self):
        return f'BCGDataset(split={self.split_name!r}, selector={self._format_years()!r}, graphs={len(self)}, feature_mode={self.feature_mode!r}, time_field={self.time_field!r})'


def _manifest_path_for_config(output_dir, setting, seed, train_ratio, val_ratio, test_ratio, min_family_samples, min_families_per_type, min_type_samples):
    if not output_dir:
        return None
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    filename = f"bcg_{setting}_seed_{seed}_r{train_ratio:.2f}-{val_ratio:.2f}-{test_ratio:.2f}_fammin{min_family_samples}_typefammin{min_families_per_type}_typemin{min_type_samples}.json"
    return os.path.join(output_dir, filename)


def _try_load_split_manifest(all_dataset, manifest_path, expected_setting, expected_seed):
    if not manifest_path or not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('implementation') != BCG_SPLIT_IMPLEMENTATION:
        return None
    if str(manifest.get('setting')) != str(expected_setting) or int(manifest.get('seed', -1)) != int(expected_seed):
        return None
    split_sha256 = manifest.get("split_sha256")
    if not isinstance(split_sha256, dict) or any((name not in split_sha256 for name in ('train', 'val', 'test'))):
        return None
    sha_to_index = {str(metadata["sha256"]): index for index, metadata in enumerate(all_dataset.graph_metadata)}
    try: split_indices = {name: [sha_to_index[str(sha)] for sha in split_sha256[name]] for name in ("train", "val", "test")}
    except KeyError: return None
    if any((not split_indices[name] for name in ('train', 'val', 'test'))):
        return None
    print(f"Reusing cached BCG split manifest: {manifest_path}")
    return split_indices, manifest


def build_bcg_family_ood_datasets(bcg_root, setting, feature_mode="ldp", time_field="vt_year", seed=0, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, min_family_samples=1, min_families_per_type=3, min_type_samples=1, split_manifest_dir=None):
    aliases = {"intra_type": "intra_type_family", "intra-type": "intra_type_family", "intra_type_family": "intra_type_family", "cross_type": "cross_type_family", "cross-type": "cross_type_family", "cross_type_family": "cross_type_family"}
    setting = aliases.get(str(setting).lower())
    if setting is None:
        raise ValueError("Unsupported BCG family-OOD setting. Expected 'intra_type_family' or 'cross_type_family'.")
    train_ratio, val_ratio, test_ratio = _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    seed, min_family_samples, min_families_per_type, min_type_samples = int(seed), int(min_family_samples), int(min_families_per_type), int(min_type_samples)
    if seed < 0:
        raise ValueError(f'BCG split seed must be non-negative, got {seed}.')
    if min_family_samples < 1 or min_families_per_type < 1 or min_type_samples < 1:
        raise ValueError('BCG minimum-count thresholds must all be at least 1.')

    all_dataset = _get_shared_bcg_dataset(bcg_root, feature_mode, time_field)
    manifest_path = _manifest_path_for_config(split_manifest_dir, setting, seed, train_ratio, val_ratio, test_ratio, min_family_samples, min_families_per_type, min_type_samples)
    cached = _try_load_split_manifest(all_dataset, manifest_path, setting, seed)
    if cached is not None:
        split_indices, split_info = cached
        validation = _validate_family_ood_split(all_dataset, setting, split_indices)
        split_info["validation"] = validation
        split_info["manifest_path"] = manifest_path
        train = BCGDataset(bcg_root, f"{setting}_train", feature_mode, time_field, indices=split_indices["train"])
        val = BCGDataset(bcg_root, f"{setting}_val", feature_mode, time_field, indices=split_indices["val"])
        test = BCGDataset(bcg_root, f"{setting}_test", feature_mode, time_field, indices=split_indices["test"])
        _print_family_ood_summary(split_info)
        return train, val, test, split_info
    benign_indices, malware_indices, missing_family_indices, missing_type_indices = [], [], [], []
    family_to_indices = defaultdict(list)
    family_to_type_counts = defaultdict(lambda: defaultdict(int))
    family_to_type_indices = defaultdict(lambda: defaultdict(list))
    family_to_year_counts = defaultdict(lambda: defaultdict(int))

    for index, metadata in enumerate(all_dataset.graph_metadata):
        label = int(metadata["label"])
        if label == 0:
            benign_indices.append(index)
            continue
        if label != 1:
            raise RuntimeError(f'Unexpected BCG binary label {label} at index {index}.')
        malware_indices.append(index)
        family, malware_type = metadata.get("family"), metadata.get("malware_type")
        if not family:
            missing_family_indices.append(index)
        if not malware_type:
            missing_type_indices.append(index)
        if not family or not malware_type:
            continue
        family, malware_type, year = str(family), str(malware_type), int(metadata["year"])
        family_to_indices[family].append(index)
        family_to_type_counts[family][malware_type] += 1
        family_to_type_indices[family][malware_type].append(index)
        family_to_year_counts[family][year] += 1

    multi_type_families = {family for family, counts in family_to_type_counts.items() if len(counts) > 1}
    benign_split = _split_individual_indices(benign_indices, train_ratio, val_ratio, test_ratio, seed)

    if setting == "intra_type_family":
        eligible_families = {family for family, indices in family_to_indices.items() if len(indices) >= min_family_samples}
        if len(eligible_families) < 3:
            raise RuntimeError('Intra-Type Family-OOD needs at least three eligible malware families.')
        family_indices = {family: list(family_to_indices[family]) for family in eligible_families}
        type_counts = {family: dict(family_to_type_counts[family]) for family in eligible_families}
        year_counts = {family: dict(family_to_year_counts[family]) for family in eligible_families}
        type_family_support = defaultdict(int)
        for family in eligible_families:
            for malware_type in type_counts[family]:
                type_family_support[malware_type] += 1
        low_support_types = sorted(malware_type for malware_type, count in type_family_support.items() if count < min_families_per_type)
        profiles = {family: _merge_profile_counts(("type", type_counts[family]), ("year", year_counts[family])) for family in eligible_families}
        malware_split, family_groups = _type_aware_family_split(family_indices, type_counts, profiles, train_ratio, val_ratio, test_ratio, seed)
        selected_types = sorted({malware_type for family in eligible_families for malware_type in type_counts[family]})
        setting_details = {"eligible_families": len(eligible_families), "multi_type_families_kept": len(multi_type_families & eligible_families), "low_support_types": low_support_types, "family_groups": {name: sorted(groups) for name, groups in family_groups.items()}}
    else:
        malware_split, type_groups, setting_details = _build_cross_type_family_split(all_dataset, family_to_indices, family_to_type_indices, train_ratio, val_ratio, test_ratio, seed, min_family_samples, min_type_samples)
        selected_types = sorted(set(type_groups["train"]) | set(type_groups["val"]) | set(type_groups["test"]))

    split_indices = {}
    for split_name in ("train", "val", "test"):
        combined = list(benign_split[split_name]) + list(malware_split[split_name])
        rng = random.Random(seed * 1009 + {"train": 11, "val": 17, "test": 23}[split_name])
        rng.shuffle(combined)
        split_indices[split_name] = combined

    validation = _validate_family_ood_split(all_dataset, setting, split_indices)
    train = BCGDataset(bcg_root, f"{setting}_train", feature_mode, time_field, indices=split_indices["train"])
    val = BCGDataset(bcg_root, f"{setting}_val", feature_mode, time_field, indices=split_indices["val"])
    test = BCGDataset(bcg_root, f"{setting}_test", feature_mode, time_field, indices=split_indices["test"])

    used_malware = set().union(*(set(malware_split[name]) for name in ("train", "val", "test")))
    split_info = {
        "implementation": BCG_SPLIT_IMPLEMENTATION,
        "setting": setting,
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "thresholds": {"min_family_samples": min_family_samples, "min_families_per_type": min_families_per_type, "min_type_samples": min_type_samples},
        "raw_counts": {"all_graphs": len(all_dataset), "benign": len(benign_indices), "malware": len(malware_indices), "families_with_family_and_type": len(family_to_indices), "multi_type_families": len(multi_type_families), "malware_missing_family": len(missing_family_indices), "malware_missing_type": len(missing_type_indices), "excluded_malware": len(set(malware_indices) - used_malware)},
        "selected_types": selected_types,
        "setting_details": setting_details,
        "validation": validation,
        "splits": {name: _describe_split(all_dataset, split_indices[name]) for name in ("train", "val", "test")},
    }
    if split_manifest_dir:
        split_info['manifest_path'] = _write_split_manifest(all_dataset, split_indices, split_info, split_manifest_dir)
    _print_family_ood_summary(split_info)
    return train, val, test, split_info


def _type_aware_family_split(group_to_indices, family_to_type_counts, group_to_profile_counts, train_ratio, val_ratio, test_ratio, seed):
    best = None
    for trial in range(128):
        split_indices, split_groups = _profile_balanced_group_split(group_to_indices, group_to_profile_counts, train_ratio, val_ratio, test_ratio, int(seed) * 1000003 + trial, trials=1)
        try: repaired_indices, repaired_groups = _repair_intra_train_type_coverage(split_indices, split_groups, group_to_indices, family_to_type_counts)
        except RuntimeError: continue
        score = _final_intra_split_score(repaired_indices, group_to_profile_counts, repaired_groups, train_ratio, val_ratio, test_ratio)
        if best is None or score < best[0]:
            best = (score, repaired_indices, repaired_groups)
    if best is None:
        raise RuntimeError('Unable to construct a valid Intra-Type Family-OOD split.')
    return best[1], best[2]


def _repair_intra_train_type_coverage(split_indices, split_groups, group_to_indices, family_to_type_counts):
    split_indices = {name: list(values) for name, values in split_indices.items()}
    split_groups = {name: list(values) for name, values in split_groups.items()}
    while True:
        train_types = {malware_type for family in split_groups["train"] for malware_type in family_to_type_counts[family]}
        eval_types = {malware_type for name in ("val", "test") for family in split_groups[name] for malware_type in family_to_type_counts[family]}
        missing = eval_types - train_types
        if not missing:
            break
        candidates = []
        for name in ("val", "test"):
            if len(split_groups[name]) <= 1:
                continue
            for family in split_groups[name]:
                covered = missing & set(family_to_type_counts[family])
                if covered:
                    candidates.append((-len(covered), len(group_to_indices[family]), name, family))
        if not candidates:
            raise RuntimeError(f'Unable to repair Intra-Type type coverage. Missing types: {sorted(missing)}')
        _, _, source, family = min(candidates)
        split_groups[source].remove(family)
        split_groups["train"].append(family)
        moved = set(group_to_indices[family])
        split_indices[source] = [index for index in split_indices[source] if index not in moved]
        split_indices["train"].extend(group_to_indices[family])
    if any((not split_groups[name] or not split_indices[name] for name in ('train', 'val', 'test'))):
        raise RuntimeError('Intra-Type repair produced an empty split.')
    return split_indices, split_groups


def _final_intra_split_score(split_indices, group_to_profile_counts, split_groups, train_ratio, val_ratio, test_ratio):
    ratios = {"train": float(train_ratio), "val": float(val_ratio), "test": float(test_ratio)}
    total = sum(len(values) for values in split_indices.values())
    sample_error = sum(((len(split_indices[name]) - ratios[name] * total) / max(ratios[name] * total, 1.0)) ** 2 for name in ratios)
    totals = defaultdict(int)
    for counts in group_to_profile_counts.values():
        for key, value in counts.items():
            totals[key] += int(value)
    split_profiles = {name: defaultdict(int) for name in ratios}
    for name in ratios:
        for group in split_groups[name]:
            for key, value in group_to_profile_counts[group].items():
                split_profiles[name][key] += int(value)
    error, dims = 0.0, 0
    for key, total_value in totals.items():
        if total_value <= 0:
            continue
        dims += 1
        for name in ratios:
            target = ratios[name] * total_value
            error += ((split_profiles[name].get(key, 0) - target) / max(target, 1.0)) ** 2
    return sample_error + error / max(dims, 1)


def _build_cross_type_family_split(all_dataset, family_to_indices, family_to_type_indices, train_ratio, val_ratio, test_ratio, seed, min_family_samples, min_type_samples):
    family_records, type_to_indices = {}, defaultdict(list)
    small_family_graphs = 0
    for family, all_indices in family_to_indices.items():
        if len(all_indices) < min_family_samples:
            small_family_graphs += len(all_indices)
            continue
        type_map = {str(malware_type): list(indices) for malware_type, indices in family_to_type_indices[family].items() if indices}
        if not type_map:
            continue
        family_records[str(family)] = type_map
        for malware_type, indices in type_map.items():
            type_to_indices[malware_type].extend(indices)

    eligible_types = {malware_type for malware_type, indices in type_to_indices.items() if len(indices) >= min_type_samples}
    small_type_graphs = sum(len(indices) for malware_type, indices in type_to_indices.items() if malware_type not in eligible_types)
    if len(eligible_types) < 3:
        raise RuntimeError(f'Cross-Type Family-OOD needs at least three eligible malware types, found {len(eligible_types)}.')
    family_records = {family: {malware_type: indices for malware_type, indices in type_map.items() if malware_type in eligible_types} for family, type_map in family_records.items()}
    family_records = {family: type_map for family, type_map in family_records.items() if type_map}

    type_names = sorted(eligible_types)
    minimum_counts = _resolve_cross_type_minimum_type_counts(len(type_names))
    type_counts = _allocate_type_counts(len(type_names), {"train": train_ratio, "val": val_ratio, "test": test_ratio}, minimum_counts)
    candidates = _generate_cross_type_partitions(type_names, type_counts, seed, max_trials=512)
    best = None
    rejected = defaultdict(int)

    for trial_id, groups in enumerate(candidates):
        resolved = _resolve_family_conflicts_for_type_partition(family_records, groups, train_ratio, val_ratio, test_ratio, seed=int(seed) * 1000003 + trial_id)
        if any(not resolved["split_indices"][name] for name in ("train", "val", "test")):
            rejected["empty_malware_split"] += 1
            continue
        final_stats = _cross_type_final_stats(all_dataset, resolved["split_indices"])
        valid, reason = _validate_cross_type_candidate_stats(final_stats, minimum_counts)
        if not valid:
            rejected[reason] += 1
            continue
        score, diagnostics = _score_cross_type_candidate(resolved["split_indices"], all_dataset, train_ratio, val_ratio, test_ratio, resolved["retained_graphs"], resolved["usable_graphs"], groups, final_stats)
        candidate = (score, trial_id, groups, resolved, diagnostics, final_stats)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None:
        raise RuntimeError("Unable to construct a valid Cross-Type Family-OOD split after final retained-data constraints. Rejections: " + ", ".join(f"{key}={value}" for key, value in sorted(rejected.items())))

    score, trial_id, groups, resolved, diagnostics, final_stats = best
    details = {
        "eligible_families": len(family_records),
        "eligible_types": len(type_names),
        "minimum_type_counts": minimum_counts,
        "assigned_type_counts": type_counts,
        "assigned_type_groups": {name: sorted(groups[name]) for name in ("train", "val", "test")},
        "actual_type_counts": {name: len(final_stats[name]["types"]) for name in ("train", "val", "test")},
        "actual_family_counts": {name: len(final_stats[name]["families"]) for name in ("train", "val", "test")},
        "actual_malware_counts": {name: final_stats[name]["malware"] for name in ("train", "val", "test")},
        "actual_malware_fractions": {name: final_stats[name]["fraction"] for name in ("train", "val", "test")},
        "family_conflicts": resolved["family_conflicts"],
        "conflicting_families": resolved["conflicting_families"],
        "dropped_conflict_graphs": resolved["dropped_conflict_graphs"],
        "small_family_graphs_dropped": small_family_graphs,
        "small_type_graphs_dropped": small_type_graphs,
        "retained_malware_graphs": resolved["retained_graphs"],
        "usable_malware_graphs": resolved["usable_graphs"],
        "retention_ratio": resolved["retained_graphs"] / max(resolved["usable_graphs"], 1),
        "candidate_score": score,
        "candidate_trial": trial_id,
        "ratio_error": diagnostics["ratio_error"],
        "year_error": diagnostics["year_error"],
        "type_count_error": diagnostics["type_count_error"],
        "rejected_candidates": dict(sorted(rejected.items())),
    }
    return resolved["split_indices"], details["assigned_type_groups"], details

def _cross_family_owner_objective(fixed_indices, conflict_specs, owners, ratios, usable):
    counts = {name: len(fixed_indices[name]) for name in ("train", "val", "test")}
    retained = sum(counts.values())
    for family, contributions, _ in conflict_specs:
        owner = owners[family]
        value = len(contributions[owner])
        counts[owner] += value
        retained += value
    if retained <= 0:
        return float('inf')
    relative_ratio_error = sum((((counts[name] / retained) - ratios[name]) / max(ratios[name], 1e-12)) ** 2 for name in ("train", "val", "test"))
    retention_penalty = 1.0 - retained / max(usable, 1)
    scarcity_penalty = max(0.0, 0.05 - counts["val"] / retained) ** 2 / (0.05 ** 2) + max(0.0, 0.10 - counts["test"] / retained) ** 2 / (0.10 ** 2) + max(0.0, counts["train"] / retained - 0.85) ** 2 / (0.15 ** 2)
    return 6.0 * relative_ratio_error + 1.0 * retention_penalty + 8.0 * scarcity_penalty

def _cross_type_final_stats(all_dataset, split_indices):
    stats = {}
    total = sum(len(split_indices[name]) for name in ("train", "val", "test"))
    for name in ("train", "val", "test"):
        families, types = set(), set()
        for index in split_indices[name]:
            metadata = all_dataset.metadata_at(index)
            if int(metadata['label']) != 1:
                continue
            if metadata.get('family'):
                families.add(str(metadata['family']))
            if metadata.get('malware_type'):
                types.add(str(metadata['malware_type']))
        malware = len(split_indices[name])
        stats[name] = {"malware": malware, "fraction": malware / max(total, 1), "families": families, "types": types}
    return stats

def _validate_cross_type_candidate_stats(stats, minimum_type_counts):
    fractions = {name: stats[name]["fraction"] for name in ("train", "val", "test")}
    actual_types = {name: len(stats[name]["types"]) for name in ("train", "val", "test")}
    actual_families = {name: len(stats[name]["families"]) for name in ("train", "val", "test")}
    required_types = {"train": min(3, int(minimum_type_counts["train"])), "val": min(2, int(minimum_type_counts["val"])), "test": min(2, int(minimum_type_counts["test"]))}
    for name in ("train", "val", "test"):
        if actual_types[name] < required_types[name]:
            return (False, f'actual_{name}_types')
    if actual_families['val'] < 3:
        return (False, 'actual_val_families')
    if actual_families['test'] < 3:
        return (False, 'actual_test_families')
    if fractions['train'] > 0.85:
        return (False, 'train_fraction_high')
    if fractions['val'] < 0.05:
        return (False, 'val_fraction_low')
    if fractions['test'] < 0.1:
        return (False, 'test_fraction_low')
    return True, "ok"

def _resolve_cross_type_minimum_type_counts(num_types):
    if num_types >= 7:
        return {'train': 3, 'val': 2, 'test': 2}
    if num_types == 6:
        return {'train': 2, 'val': 2, 'test': 2}
    if num_types == 5:
        return {'train': 2, 'val': 1, 'test': 2}
    if num_types == 4:
        return {'train': 2, 'val': 1, 'test': 1}
    if num_types == 3:
        return {'train': 1, 'val': 1, 'test': 1}
    raise ValueError("At least three malware types are required.")


def _allocate_type_counts(num_types, ratios, minimum_counts):
    counts = dict(minimum_counts)
    remaining = int(num_types) - sum(counts.values())
    while remaining > 0:
        chosen = max(("train", "val", "test"), key=lambda name: (((float(ratios[name]) * num_types - counts[name]) / max(float(ratios[name]) * num_types, 1.0)), -counts[name], float(ratios[name])))
        counts[chosen] += 1
        remaining -= 1
    return counts

def _generate_cross_type_partitions(type_names, counts, seed, max_trials=8192):
    type_names = list(type_names)
    train_count, val_count = int(counts["train"]), int(counts["val"])
    seen, results = set(), []
    rng = random.Random(int(seed) * 15485863 + 97)
    for _ in range(max(1, int(max_trials))):
        shuffled = list(type_names)
        rng.shuffle(shuffled)
        train_end, val_end = train_count, train_count + val_count
        groups = {"train": tuple(sorted(shuffled[:train_end])), "val": tuple(sorted(shuffled[train_end:val_end])), "test": tuple(sorted(shuffled[val_end:]))}
        key = (groups["train"], groups["val"], groups["test"])
        if key not in seen:
            seen.add(key)
            results.append(groups)
    if len(type_names) <= 10:
        for train_types in itertools.combinations(sorted(type_names), train_count):
            remaining = tuple(value for value in sorted(type_names) if value not in set(train_types))
            for val_types in itertools.combinations(remaining, val_count):
                val_set = set(val_types)
                groups = {"train": tuple(sorted(train_types)), "val": tuple(sorted(val_types)), "test": tuple(value for value in remaining if value not in val_set)}
                key = (groups["train"], groups["val"], groups["test"])
                if key not in seen:
                    seen.add(key)
                    results.append(groups)
    if not results:
        raise RuntimeError('No Cross-Type type partitions were generated.')
    return results


def _resolve_family_conflicts_for_type_partition(family_records, type_groups, train_ratio, val_ratio, test_ratio, seed=0):
    ratios = {"train": float(train_ratio), "val": float(val_ratio), "test": float(test_ratio)}
    type_owner = {}
    for name in ("train", "val", "test"):
        for malware_type in type_groups[name]:
            if malware_type in type_owner:
                raise RuntimeError(f'Malware type {malware_type!r} assigned to multiple splits.')
            type_owner[malware_type] = name

    fixed_indices = {"train": [], "val": [], "test": []}
    conflict_specs, usable = [], 0
    for family, type_map in family_records.items():
        contributions = {"train": [], "val": [], "test": []}
        for malware_type, indices in type_map.items():
            owner = type_owner.get(malware_type)
            if owner is None:
                continue
            contributions[owner].extend(indices)
            usable += len(indices)
        nonempty = [name for name in ("train", "val", "test") if contributions[name]]
        if not nonempty:
            continue
        if len(nonempty) == 1: fixed_indices[nonempty[0]].extend(contributions[nonempty[0]])
        else: conflict_specs.append((family, contributions, nonempty))

    fixed_counts = {name: len(fixed_indices[name]) for name in ("train", "val", "test")}
    rng = random.Random(int(seed))
    num_restarts = 4
    best = None

    for restart in range(num_restarts):
        if restart == 0:
            owners = {family: min(nonempty, key=lambda name: (-len(contributions[name]), {"train": 0, "val": 1, "test": 2}[name])) for family, contributions, nonempty in conflict_specs}
        else:
            owners = {family: rng.choice(nonempty) for family, _, nonempty in conflict_specs}

        counts = dict(fixed_counts)
        retained = sum(counts.values())
        for family, contributions, _ in conflict_specs:
            owner = owners[family]
            value = len(contributions[owner])
            counts[owner] += value
            retained += value

        improved = True
        iterations = 0
        while improved and iterations < 4:
            improved = False
            iterations += 1
            order = list(conflict_specs)
            rng.shuffle(order)
            for family, contributions, nonempty in order:
                current = owners[family]
                current_value = len(contributions[current])
                local_best, local_best_score = current, _cross_owner_counts_objective(counts, retained, ratios, usable)
                for candidate_owner in nonempty:
                    if candidate_owner == current:
                        continue
                    candidate_value = len(contributions[candidate_owner])
                    trial_counts = dict(counts)
                    trial_counts[current] -= current_value
                    trial_counts[candidate_owner] += candidate_value
                    trial_retained = retained - current_value + candidate_value
                    score = _cross_owner_counts_objective(trial_counts, trial_retained, ratios, usable)
                    if score + 1e-12 < local_best_score:
                        local_best, local_best_score = candidate_owner, score
                if local_best != current:
                    new_value = len(contributions[local_best])
                    counts[current] -= current_value
                    counts[local_best] += new_value
                    retained = retained - current_value + new_value
                    owners[family] = local_best
                    improved = True

        split_indices = {name: list(fixed_indices[name]) for name in ("train", "val", "test")}
        dropped = 0
        for family, contributions, nonempty in conflict_specs:
            owner = owners[family]
            split_indices[owner].extend(contributions[owner])
            dropped += sum(len(contributions[name]) for name in nonempty if name != owner)
        for name in ('train', 'val', 'test'):
            split_indices[name] = sorted(set(split_indices[name]))
        retained = sum(len(split_indices[name]) for name in ("train", "val", "test"))
        objective = _cross_owner_counts_objective({name: len(split_indices[name]) for name in ("train", "val", "test")}, retained, ratios, usable)
        candidate = (objective, -retained, split_indices, dropped)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    _, neg_retained, split_indices, dropped = best
    return {"split_indices": split_indices, "family_conflicts": len(conflict_specs), "conflicting_families": sorted(family for family, _, _ in conflict_specs), "dropped_conflict_graphs": dropped, "retained_graphs": -neg_retained, "usable_graphs": usable}


def _cross_owner_counts_objective(counts, retained, ratios, usable):
    if retained <= 0:
        return float('inf')
    relative_ratio_error = sum((((counts[name] / retained) - ratios[name]) / max(ratios[name], 1e-12)) ** 2 for name in ("train", "val", "test"))
    retention_penalty = 1.0 - retained / max(usable, 1)
    scarcity_penalty = max(0.0, 0.05 - counts["val"] / retained) ** 2 / (0.05 ** 2) + max(0.0, 0.10 - counts["test"] / retained) ** 2 / (0.10 ** 2) + max(0.0, counts["train"] / retained - 0.85) ** 2 / (0.15 ** 2)
    return 6.0 * relative_ratio_error + 1.0 * retention_penalty + 8.0 * scarcity_penalty


def _score_cross_type_candidate(split_indices, all_dataset, train_ratio, val_ratio, test_ratio, retained_graphs, usable_graphs, type_groups, final_stats=None):
    ratios = {"train": float(train_ratio), "val": float(val_ratio), "test": float(test_ratio)}
    total = sum(len(split_indices[name]) for name in ratios)
    if total <= 0:
        return (float('inf'), {'ratio_error': float('inf'), 'year_error': float('inf'), 'type_count_error': float('inf')})
    observed = {name: len(split_indices[name]) / total for name in ratios}
    ratio_error = sum(((observed[name] - ratios[name]) / max(ratios[name], 1e-12)) ** 2 for name in ratios)

    year_totals, split_years = defaultdict(int), {name: defaultdict(int) for name in ratios}
    for name in ratios:
        for index in split_indices[name]:
            year = int(all_dataset.metadata_at(index)["year"])
            year_totals[year] += 1
            split_years[name][year] += 1
    year_error, dims = 0.0, 0
    for year, year_total in year_totals.items():
        if year_total <= 0:
            continue
        dims += 1
        global_rate = year_total / total
        for name in ratios:
            year_error += (split_years[name].get(year, 0) / max(len(split_indices[name]), 1) - global_rate) ** 2
    year_error /= max(dims, 1)

    if final_stats is None:
        final_stats = _cross_type_final_stats(all_dataset, split_indices)
    actual_type_counts = {name: len(final_stats[name]["types"]) for name in ratios}
    total_actual_types = sum(actual_type_counts.values())
    type_count_error = sum(((actual_type_counts[name] / max(total_actual_types, 1)) - ratios[name]) ** 2 for name in ratios)
    retention_penalty = 1.0 - retained_graphs / max(usable_graphs, 1)
    score = 5.0 * ratio_error + 1.5 * retention_penalty + 0.5 * year_error + 0.25 * type_count_error
    return score, {"ratio_error": ratio_error, "year_error": year_error, "type_count_error": type_count_error}

def _profile_balanced_group_split(group_to_indices, group_to_profile_counts, train_ratio, val_ratio, test_ratio, seed, trials=64):
    ratios = {"train": float(train_ratio), "val": float(val_ratio), "test": float(test_ratio)}
    groups = [str(group) for group, indices in group_to_indices.items() if indices]
    if len(groups) < 3:
        raise ValueError('At least three non-empty groups are required.')
    best = None
    for trial in range(max(1, int(trials))):
        rng = random.Random(int(seed) * 1000003 + trial)
        shuffled = list(groups)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda group: len(group_to_indices[group]), reverse=True)
        total = sum(len(group_to_indices[group]) for group in shuffled)
        sample_targets = {name: ratios[name] * total for name in ratios}
        profile_totals = defaultdict(int)
        for group in shuffled:
            for key, value in group_to_profile_counts.get(group, {}).items():
                profile_totals[key] += int(value)
        profile_targets = {name: {key: ratios[name] * value for key, value in profile_totals.items()} for name in ratios}
        assigned, sample_counts = {name: [] for name in ratios}, {name: 0 for name in ratios}
        profile_counts = {name: defaultdict(int) for name in ratios}
        seed_order = sorted(ratios, key=lambda name: ratios[name], reverse=True)
        for group, name in zip(shuffled[:3], seed_order):
            _assign_group(group, name, assigned, sample_counts, profile_counts, group_to_indices, group_to_profile_counts)
        for group in shuffled[3:]:
            noise = {name: rng.random() for name in ratios}
            chosen = min(ratios, key=lambda name: (_candidate_profile_score(name, group, ratios, sample_targets, profile_targets, sample_counts, profile_counts, group_to_indices, group_to_profile_counts), noise[name]))
            _assign_group(group, chosen, assigned, sample_counts, profile_counts, group_to_indices, group_to_profile_counts)
        indices = {name: [index for group in assigned[name] for index in group_to_indices[group]] for name in ratios}
        score = _completed_profile_score(sample_counts, profile_counts, ratios, sample_targets, profile_targets)
        if best is None or score < best[0]:
            best = (score, indices, assigned)
    return best[1], best[2]


def _assign_group(group, split_name, assigned, sample_counts, profile_counts, group_to_indices, group_to_profile_counts):
    assigned[split_name].append(group)
    sample_counts[split_name] += len(group_to_indices[group])
    for key, value in group_to_profile_counts.get(group, {}).items():
        profile_counts[split_name][key] += int(value)


def _candidate_profile_score(candidate, group, ratios, sample_targets, profile_targets, sample_counts, profile_counts, group_to_indices, group_to_profile_counts):
    trial_samples = dict(sample_counts)
    trial_samples[candidate] += len(group_to_indices[group])
    sample_error = sum(((trial_samples[name] - sample_targets[name]) / max(sample_targets[name], 1.0)) ** 2 for name in ratios)
    group_profile = group_to_profile_counts.get(group, {})
    error, dims = 0.0, 0
    keys = set().union(*(targets.keys() for targets in profile_targets.values()))
    for key in keys:
        dims += 1
        for name in ratios:
            value = profile_counts[name].get(key, 0) + (int(group_profile.get(key, 0)) if name == candidate else 0)
            target = profile_targets[name].get(key, 0.0)
            error += ((value - target) / max(target, 1.0)) ** 2
    return sample_error + error / max(dims, 1)


def _completed_profile_score(sample_counts, profile_counts, ratios, sample_targets, profile_targets):
    sample_error = sum(((sample_counts[name] - sample_targets[name]) / max(sample_targets[name], 1.0)) ** 2 for name in ratios)
    error, dims = 0.0, 0
    keys = set().union(*(targets.keys() for targets in profile_targets.values()))
    for key in keys:
        dims += 1
        for name in ratios:
            target = profile_targets[name].get(key, 0.0)
            error += ((profile_counts[name].get(key, 0) - target) / max(target, 1.0)) ** 2
    return sample_error + error / max(dims, 1)


def _merge_profile_counts(*named_counts):
    merged = {}
    for prefix, counts in named_counts:
        for key, value in counts.items():
            merged[f'{prefix}::{key}'] = int(value)
    return merged


def _split_individual_indices(indices, train_ratio, val_ratio, test_ratio, seed):
    indices = list(indices)
    if len(indices) < 3:
        raise ValueError('At least three benign graphs are required.')
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    n = len(indices)
    n_train, n_val = max(1, int(round(n * train_ratio))), max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        overflow = n_train + n_val - (n - 1)
        reduce_train = min(overflow, max(0, n_train - 1))
        n_train -= reduce_train
        overflow -= reduce_train
        if overflow > 0:
            n_val = max(1, n_val - overflow)
    train, val, test = indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]
    if not train or not val or (not test):
        raise RuntimeError('Benign split produced an empty partition.')
    return {"train": train, "val": val, "test": test}


def _validate_split_ratios(train_ratio, val_ratio, test_ratio):
    values = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if any((value <= 0.0 for value in values)):
        raise ValueError(f'BCG split ratios must be positive, got {values}.')
    if abs(sum(values) - 1.0) > 1e-08:
        raise ValueError(f'BCG train/val/test ratios must sum to 1.0, got {values}.')
    return tuple(values)


def _validate_family_ood_split(all_dataset, setting, split_indices):
    sets = {name: set(values) for name, values in split_indices.items()}
    if sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test']:
        raise RuntimeError('BCG graph leakage detected across splits.')
    descriptions = {name: _describe_split(all_dataset, values) for name, values in split_indices.items()}
    families = {name: set(descriptions[name]["families"]) for name in descriptions}
    types = {name: set(descriptions[name]["types"]) for name in descriptions}
    family_overlaps = {"train_val": sorted(families["train"] & families["val"]), "train_test": sorted(families["train"] & families["test"]), "val_test": sorted(families["val"] & families["test"])}
    if any(family_overlaps.values()):
        raise RuntimeError(f'BCG family leakage detected: {family_overlaps}')
    type_overlaps = {"train_val": sorted(types["train"] & types["val"]), "train_test": sorted(types["train"] & types["test"]), "val_test": sorted(types["val"] & types["test"])}
    if setting == "intra_type_family":
        unseen_val, unseen_test = sorted(types["val"] - types["train"]), sorted(types["test"] - types["train"])
        if unseen_val or unseen_test:
            raise RuntimeError(f'Intra-Type contains evaluation types absent from training. val_only={unseen_val}, test_only={unseen_test}')
    elif setting == "cross_type_family":
        if any(type_overlaps.values()):
            raise RuntimeError(f'Cross-Type malware-type leakage detected: {type_overlaps}')
    else: raise ValueError(f"Unknown BCG setting: {setting}")
    for name, description in descriptions.items():
        if description['benign'] < 1 or description['malware'] < 1:
            raise RuntimeError(f'BCG {name} must contain both classes.')
    return {"family_overlap_counts": {key: len(value) for key, value in family_overlaps.items()}, "type_overlap_counts": {key: len(value) for key, value in type_overlaps.items()}, "unseen_val_types": sorted(types["val"] - types["train"]), "unseen_test_types": sorted(types["test"] - types["train"])}


def _describe_split(all_dataset, indices):
    benign = malware = 0
    families, types, years, type_counts = set(), set(), defaultdict(int), defaultdict(int)
    for index in indices:
        metadata = all_dataset.metadata_at(index)
        label = int(metadata["label"])
        years[int(metadata["year"])] += 1
        if label == 0: benign += 1
        elif label == 1:
            malware += 1
            if metadata.get('family'):
                families.add(str(metadata['family']))
            if metadata.get("malware_type"):
                malware_type = str(metadata["malware_type"])
                types.add(malware_type)
                type_counts[malware_type] += 1
        else: raise RuntimeError(f"Unexpected BCG label: {label}")
    return {"graphs": benign + malware, "benign": benign, "malware": malware, "families": sorted(families), "types": sorted(types), "type_counts": dict(sorted(type_counts.items())), "years": {str(year): count for year, count in sorted(years.items())}}


def _write_split_manifest(all_dataset, split_indices, split_info, output_dir):
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    thresholds, ratio = split_info["thresholds"], split_info["ratios"]
    filename = f"bcg_{split_info['setting']}_seed_{split_info['seed']}_r{ratio['train']:.2f}-{ratio['val']:.2f}-{ratio['test']:.2f}_fammin{thresholds['min_family_samples']}_typefammin{thresholds['min_families_per_type']}_typemin{thresholds['min_type_samples']}.json"
    path = os.path.join(output_dir, filename)
    manifest = dict(split_info)
    manifest["split_sha256"] = {name: [str(all_dataset.metadata_at(index)["sha256"]) for index in split_indices[name]] for name in ("train", "val", "test")}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def _print_family_ood_summary(split_info):
    print("\n" + "=" * 92)
    print("BCG Family-OOD split")
    print(f"Implementation: {split_info['implementation']}")
    print(f"Setting: {split_info['setting']}")
    print(f"Split seed: {split_info['seed']}")
    print(f"Target ratio: {split_info['ratios']['train']:.2f}/{split_info['ratios']['val']:.2f}/{split_info['ratios']['test']:.2f}")
    print("-" * 92)
    for name in ("train", "val", "test"):
        d = split_info["splits"][name]
        print(f"{name:>5} | graphs={d['graphs']:5d} | benign={d['benign']:5d} | malware={d['malware']:5d} | families={len(d['families']):3d} | types={len(d['types']):3d}")
    validation = split_info["validation"]
    fov, tov = validation["family_overlap_counts"], validation["type_overlap_counts"]
    print("-" * 92)
    print(f"Family overlap | train-val={fov['train_val']} | train-test={fov['train_test']} | val-test={fov['val_test']}")
    print(f"Type overlap   | train-val={tov['train_val']} | train-test={tov['train_test']} | val-test={tov['val_test']}")
    raw, details = split_info["raw_counts"], split_info["setting_details"]
    print(f"Raw metadata   | malware={raw['malware']} | families={raw['families_with_family_and_type']} | multi-type families={raw['multi_type_families']} | missing family graphs={raw['malware_missing_family']} | missing type graphs={raw['malware_missing_type']}")
    if split_info["setting"] == "intra_type_family":
        print(f"Intra-Type     | eligible families={details['eligible_families']} | multi-type families kept={details['multi_type_families_kept']} | low-support types(<{split_info['thresholds']['min_families_per_type']} families)={len(details['low_support_types'])}")
    else:
        groups = details["assigned_type_groups"]
        actual_types, actual_families = details["actual_type_counts"], details["actual_family_counts"]
        actual_malware, actual_fractions = details["actual_malware_counts"], details["actual_malware_fractions"]
        print(f"Cross-Type     | eligible families={details['eligible_families']} | eligible types={details['eligible_types']} | assigned types={len(groups['train'])}/{len(groups['val'])}/{len(groups['test'])}")
        print(f"Actual retained| types={actual_types['train']}/{actual_types['val']}/{actual_types['test']} | families={actual_families['train']}/{actual_families['val']}/{actual_families['test']} | malware={actual_malware['train']}/{actual_malware['val']}/{actual_malware['test']}")
        print(f"Malware ratio  | train={actual_fractions['train']:.4f} | val={actual_fractions['val']:.4f} | test={actual_fractions['test']:.4f}")
        print(f"Family conflict| conflicting families={details['family_conflicts']} | conflict graphs dropped={details['dropped_conflict_graphs']} | retention={details['retention_ratio']:.4f}")
        print(f"Type groups    | train={groups['train']} | val={groups['val']} | test={groups['test']}")
    print(f"Final filtering| excluded malware graphs={raw['excluded_malware']} | retained malware graphs={raw['malware'] - raw['excluded_malware']}")
    if split_info.get('manifest_path'):
        print(f"Manifest: {split_info['manifest_path']}")
    print("=" * 92 + "\n")
