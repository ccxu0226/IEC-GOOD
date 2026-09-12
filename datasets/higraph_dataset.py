import json
import os

import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data, InMemoryDataset


class HiGraphDataset(InMemoryDataset):
    def __init__(self, higraph_root, years, split_name, feature_mode="cfg_mean"):
        self.higraph_root = os.path.abspath(higraph_root)
        self.raw_data_dir = os.path.join(self.higraph_root, "data")
        self.years = tuple(int(year) for year in years)
        self.split_name = str(split_name)
        self.feature_mode = str(feature_mode)

        if self.feature_mode not in {'constant', 'cfg_mean'}:
            raise ValueError(f'Unsupported HiGraph feature mode: {self.feature_mode}')

        self.input_dim = 1 if self.feature_mode == "constant" else 11
        self.declared_num_classes = 2
        self.large_scale = True

        processed_root = os.path.join(self.higraph_root, "iec_good_cache", f"{self.split_name}_{self.feature_mode}")

        super().__init__(root=processed_root)
        self.load(self.processed_paths[0])
        self._print_year_label_statistics()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        data_list = []
        total_rows = 0
        total_graphs = 0
        skipped_graphs = 0

        for year in self.years:
            sources = ((0, f"{year}_Benign.parquet"), (1, f"{year}_Malware_15.parquet"))

            for binary_label, filename in sources:
                parquet_path = os.path.join(self.raw_data_dir, filename)

                if not os.path.isfile(parquet_path):
                    raise FileNotFoundError(f'HiGraph parquet file not found: {parquet_path}')

                columns = ["function_count", "function_edges"]

                if self.feature_mode == 'cfg_mean':
                    columns.append('control_flow_edges')

                parquet_file = pq.ParquetFile(parquet_path)

                print(f"Processing HiGraph | year={year} | label={binary_label} | file={filename}")

                file_rows = 0
                file_graphs = 0
                file_skipped = 0

                for record_batch in parquet_file.iter_batches(batch_size=256, columns=columns):
                    rows = record_batch.to_pylist()

                    for row in rows:
                        total_rows += 1
                        file_rows += 1
                        data = self._row_to_data(row=row, binary_label=binary_label, year=year)

                        if data is None:
                            skipped_graphs += 1
                            file_skipped += 1
                            continue

                        data_list.append(data)
                        total_graphs += 1
                        file_graphs += 1

                    if file_rows % 10000 < len(rows):
                        print(f'HiGraph progress | year={year} | label={binary_label} | rows={file_rows} | graphs={file_graphs} | skipped={file_skipped} | total_graphs={total_graphs}')

                label_name = "Benign" if binary_label == 0 else "Malware"
                print(f"Finished HiGraph file | year={year} | label={binary_label} ({label_name}) | rows={file_rows} | graphs={file_graphs} | skipped={file_skipped}")

        if not data_list:
            raise RuntimeError(f"No valid HiGraph graphs were produced for split '{self.split_name}'.")

        print(f"Collating HiGraph InMemoryDataset | split={self.split_name} | graphs={total_graphs} | skipped={skipped_graphs}")
        self.save(data_list, self.processed_paths[0])
        print(f"Saved HiGraph InMemoryDataset | split={self.split_name} | graphs={total_graphs} | path={self.processed_paths[0]}")

    def _row_to_data(self, row, binary_label, year):
        num_nodes = int(row.get("function_count") or 0)

        if num_nodes <= 0:
            return None

        edge_index = self._build_edge_index(row.get("function_edges"), num_nodes)

        if edge_index is None:
            return None

        if self.feature_mode == "constant":
            x = torch.ones((num_nodes, 1), dtype=torch.float32)
        else:
            x = self._build_cfg_mean_features(raw_cfg=row.get("control_flow_edges"), num_nodes=num_nodes)

        return Data(x=x, edge_index=edge_index, y=torch.tensor([int(binary_label)], dtype=torch.long), year=torch.tensor([int(year)], dtype=torch.long))

    def _print_year_label_statistics(self):
        if len(self) == 0:
            print(f"HiGraph statistics | split={self.split_name} | dataset is empty")
            return

        if not hasattr(self._data, "y") or not hasattr(self._data, "year"):
            print(f"HiGraph statistics | split={self.split_name} | y/year information unavailable")
            return

        labels = self._data.y.view(-1).cpu()
        years = self._data.year.view(-1).cpu()

        print(f"\n{'=' * 80}")
        print(f"HiGraph dataset statistics | split={self.split_name} | feature_mode={self.feature_mode}")
        print(f"{'=' * 80}")

        total_benign = 0
        total_malware = 0

        for year in sorted(torch.unique(years).tolist()):
            year_mask = years == int(year)
            year_labels = labels[year_mask]
            benign_count = int((year_labels == 0).sum().item())
            malware_count = int((year_labels == 1).sum().item())
            total_count = int(year_labels.numel())

            total_benign += benign_count
            total_malware += malware_count

            benign_ratio = benign_count / total_count * 100 if total_count > 0 else 0.0
            malware_ratio = malware_count / total_count * 100 if total_count > 0 else 0.0

            print(f"Year {int(year)} | Total={total_count} | Benign={benign_count} ({benign_ratio:.2f}%) | Malware={malware_count} ({malware_ratio:.2f}%)")

        total_count = total_benign + total_malware
        benign_ratio = total_benign / total_count * 100 if total_count > 0 else 0.0
        malware_ratio = total_malware / total_count * 100 if total_count > 0 else 0.0

        print("-" * 80)
        print(f"Split total | Total={total_count} | Benign={total_benign} ({benign_ratio:.2f}%) | Malware={total_malware} ({malware_ratio:.2f}%)")
        print(f"{'=' * 80}\n")

    @staticmethod
    def _build_edge_index(raw_edges, num_nodes):
        if raw_edges is None:
            return torch.empty((2, 0), dtype=torch.long)

        edge_index = torch.as_tensor(raw_edges, dtype=torch.long)

        if edge_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)

        if edge_index.dim() != 2:
            return None

        if edge_index.size(0) == 2:
            pass
        elif edge_index.size(1) == 2:
            edge_index = edge_index.t()
        else:
            return None

        edge_index = edge_index.contiguous()

        if edge_index.numel() > 0:
            if int(edge_index.min()) < 0:
                return None
            if int(edge_index.max()) >= num_nodes:
                return None

        return edge_index

    @staticmethod
    def _build_cfg_mean_features(raw_cfg, num_nodes):
        feature_dim = 11
        x = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)

        if raw_cfg is None:
            return x

        if isinstance(raw_cfg, str):
            try:
                cfg_dict = json.loads(raw_cfg)
            except json.JSONDecodeError:
                return x
        elif isinstance(raw_cfg, dict):
            cfg_dict = raw_cfg
        else:
            return x

        for function_id, cfg in cfg_dict.items():
            try:
                node_index = int(function_id)
            except (TypeError, ValueError):
                continue

            if node_index < 0 or node_index >= num_nodes:
                continue
            if not isinstance(cfg, dict):
                continue

            block_features = cfg.get("block_features")

            if block_features is None:
                continue

            block_features = torch.as_tensor(block_features, dtype=torch.float32)

            if block_features.numel() == 0:
                continue
            if block_features.dim() != 2:
                continue
            if block_features.size(1) != feature_dim:
                continue

            x[node_index] = block_features.mean(dim=0)

        return x
