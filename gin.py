import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class Encoder(nn.Module):
    def __init__(self, num_features, hidden_dim, num_gc_layers):
        super().__init__()

        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_gc_layers <= 0:
            raise ValueError(f"num_gc_layers must be positive, got {num_gc_layers}.")

        self.num_features = int(num_features)
        self.hidden_dim = int(hidden_dim)
        self.num_gc_layers = int(num_gc_layers)
        self.embedding_dim = self.hidden_dim * self.num_gc_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer_index in range(self.num_gc_layers):
            input_dim = self.num_features if layer_index == 0 else self.hidden_dim
            mlp = nn.Sequential(nn.Linear(input_dim, self.hidden_dim), nn.ReLU(inplace=True), nn.Linear(self.hidden_dim, self.hidden_dim))
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_nodes(self, x, edge_index):
        x, edge_index = self._prepare_node_inputs(x=x, edge_index=edge_index)
        return self._encode_nodes_prepared(x=x, edge_index=edge_index)

    def _encode_nodes_prepared(self, x, edge_index):
        layer_representations = []

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self._apply_batch_norm(batch_norm=batch_norm, x=x)
            layer_representations.append(x)

        return torch.cat(layer_representations, dim=-1)

    def pool_nodes(self, node_representations, batch):
        if node_representations.dim() != 2:
            raise ValueError(f"node_representations must have shape [N, D], got {node_representations.shape}.")

        batch = self._prepare_batch_vector(batch=batch, num_nodes=node_representations.size(0), device=node_representations.device)
        return global_add_pool(node_representations, batch)

    def forward(self, x, edge_index, batch=None):
        if x is None:
            num_nodes = self._infer_num_nodes(edge_index=edge_index, batch=batch)
            x = torch.ones((num_nodes, self.num_features), dtype=torch.float32, device=edge_index.device)

        x, edge_index = self._prepare_node_inputs(x=x, edge_index=edge_index)
        batch = self._prepare_batch_vector(batch=batch, num_nodes=x.size(0), device=x.device)
        node_representations = self._encode_nodes_prepared(x=x, edge_index=edge_index)
        graph_representations = global_add_pool(node_representations, batch)
        return graph_representations, node_representations

    @staticmethod
    def _apply_batch_norm(batch_norm, x):
        if x.size(0) > 1 or not batch_norm.training:
            return batch_norm(x)

        return F.batch_norm(
            input=x,
            running_mean=batch_norm.running_mean,
            running_var=batch_norm.running_var,
            weight=batch_norm.weight,
            bias=batch_norm.bias,
            training=False,
            momentum=0.0,
            eps=batch_norm.eps,
        )

    def _prepare_node_inputs(self, x, edge_index):
        if x is None:
            raise ValueError("encode_nodes() requires explicit node features. Use forward() to enable automatic constant features.")

        if x.dim() == 1:
            x = x.unsqueeze(-1)

        if x.dim() != 2:
            raise ValueError(f"x must have shape [N, F], got {x.shape}.")
        if x.size(0) == 0:
            raise ValueError("Cannot encode a graph without nodes.")
        if x.size(1) != self.num_features:
            raise ValueError(f"Node-feature dimension mismatch: expected {self.num_features}, got {x.size(1)}.")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}.")

        if edge_index.device != x.device:
            edge_index = edge_index.to(x.device)

        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        if x.dtype != torch.float32:
            x = x.float()

        if not edge_index.is_contiguous():
            edge_index = edge_index.contiguous()

        return x, edge_index

    @staticmethod
    def _prepare_batch_vector(batch, num_nodes, device):
        if batch is None:
            return torch.zeros(num_nodes, dtype=torch.long, device=device)

        if batch.dim() != 1:
            raise ValueError(f"batch must have shape [N], got {batch.shape}.")
        if batch.numel() != num_nodes:
            raise ValueError(f"batch length does not match the number of nodes: {batch.numel()} != {num_nodes}.")

        if batch.device != device or batch.dtype != torch.long:
            batch = batch.to(device=device, dtype=torch.long)

        return batch

    @staticmethod
    def _infer_num_nodes(edge_index, batch):
        if batch is not None:
            if batch.dim() != 1:
                raise ValueError(f"batch must have shape [N], got {batch.shape}.")
            if batch.numel() == 0:
                raise ValueError("Cannot infer nodes from an empty batch vector.")
            return int(batch.numel())

        if edge_index.numel() == 0:
            raise ValueError("Cannot infer the number of nodes when both x and batch are absent and edge_index is empty.")

        return int(edge_index.max().item()) + 1
