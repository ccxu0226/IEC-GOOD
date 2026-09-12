from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DisturbanceRecord:
    feature_disturbed: torch.Tensor
    edge_disturbed: torch.Tensor
    edge_units: torch.Tensor


def ensure_node_features(data):
    if getattr(data, "x", None) is None:
        data.x = torch.ones(
            (int(data.num_nodes), 1),
            dtype=torch.float32,
            device=data.edge_index.device,
        )
    else:
        data.x = data.x.float()

    if data.x.dim() == 1:
        data.x = data.x.unsqueeze(-1)

    data.edge_index = data.edge_index.long().contiguous()
    return data


def evidence_aware_augment_pair(
    data,
    node_evidence,
    view_one_mask_ratio,
    view_one_edge_drop_ratio,
    view_two_mask_ratio,
    view_two_edge_drop_ratio,
    evidence_budget,
    eta,
    enabled,
):


    for name, value in (
        ("view_one_mask_ratio", view_one_mask_ratio),
        ("view_one_edge_drop_ratio", view_one_edge_drop_ratio),
        ("view_two_mask_ratio", view_two_mask_ratio),
        ("view_two_edge_drop_ratio", view_two_edge_drop_ratio),
        ("eta", eta),
    ):
        _validate_ratio(name, value)

    view_one = _clone_graph(data)
    view_two = _clone_graph(data)

    num_nodes = int(view_one.x.size(0))
    device = view_one.x.device
    dtype = view_one.x.dtype
    node_evidence = _validate_node_evidence(node_evidence, num_nodes, device, dtype)
    node_to_graph, num_graphs = _get_batch_assignment(data, num_nodes, device)
    graph_budget = _prepare_graph_budget(evidence_budget, num_graphs, device, dtype)

    if enabled:
        graph_factor = (1.0 - float(eta) * (1.0 - graph_budget)).clamp(0.0, 1.0)
        node_disturbance_weight = 1.0 - node_evidence
    else:
        graph_factor = torch.ones_like(graph_budget)
        node_disturbance_weight = torch.ones_like(node_evidence)

    node_factor = graph_factor[node_to_graph]
    disturbed_x1 = _sample_node_disturbance(
        num_nodes, view_one_mask_ratio, node_factor, node_disturbance_weight, device, dtype
    )
    disturbed_x2 = _sample_node_disturbance(
        num_nodes, view_two_mask_ratio, node_factor, node_disturbance_weight, device, dtype
    )
    view_one.x = view_one.x.masked_fill(disturbed_x1.unsqueeze(-1), 0.0)
    view_two.x = view_two.x.masked_fill(disturbed_x2.unsqueeze(-1), 0.0)

    edge_units, stored_edge_indices = _directed_non_self_edges(data.edge_index)
    edge_factor = _get_edge_budget_factor(edge_units, node_to_graph, graph_factor)
    disturbed_e1 = _sample_edge_disturbance(
        edge_units,
        node_evidence,
        view_one_edge_drop_ratio,
        edge_factor,
        enabled,
        dtype,
    )
    disturbed_e2 = _sample_edge_disturbance(
        edge_units,
        node_evidence,
        view_two_edge_drop_ratio,
        edge_factor,
        enabled,
        dtype,
    )
    _apply_directed_edge_disturbance(view_one, stored_edge_indices, disturbed_e1)
    _apply_directed_edge_disturbance(view_two, stored_edge_indices, disturbed_e2)

    return (
        view_one,
        view_two,
        DisturbanceRecord(disturbed_x1, disturbed_e1, edge_units),
        DisturbanceRecord(disturbed_x2, disturbed_e2, edge_units),
    )


def _clone_graph(data):
    try:
        cloned = data.clone()
    except (AttributeError, TypeError):
        cloned = copy.deepcopy(data)

    cloned = ensure_node_features(cloned)
    if cloned.x.dim() != 2:
        raise ValueError(f"Node attributes must have shape [N, F], got {cloned.x.shape}.")
    if int(cloned.x.size(0)) != int(cloned.num_nodes):
        raise ValueError("The number of feature rows must equal data.num_nodes.")
    return cloned


def _directed_non_self_edges(edge_index):
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}.")
    source, target = edge_index.long()
    non_self = source != target
    stored_indices = non_self.nonzero(as_tuple=False).view(-1)
    return edge_index[:, stored_indices].long(), stored_indices


def _apply_directed_edge_disturbance(data, stored_edge_indices, disturbed):
    num_edges = int(data.edge_index.size(1))
    if disturbed.dim() != 1:
        raise ValueError("edge disturbance mask must be one-dimensional.")
    if stored_edge_indices.numel() != disturbed.numel():
        raise ValueError("edge disturbance count does not match non-self-loop edges.")

    if disturbed.numel() == 0:
        return data

    keep = torch.ones(num_edges, dtype=torch.bool, device=data.edge_index.device)
    keep[stored_edge_indices] = ~disturbed
    edge_attribute_keys = _find_edge_attribute_keys(data, num_edges)
    data.edge_index = data.edge_index[:, keep].long().contiguous()
    for key in edge_attribute_keys:
        data[key] = data[key][keep]
    return data


def _sample_node_disturbance(
    num_nodes,
    base_ratio,
    budget_factor,
    disturbance_weight,
    device,
    dtype,
):
    if budget_factor.shape != (num_nodes,) or disturbance_weight.shape != (num_nodes,):
        raise ValueError("Node perturbation factors must have shape [N].")
    probability = (float(base_ratio) * budget_factor * disturbance_weight).clamp(0.0, 1.0)
    return torch.rand(num_nodes, device=device, dtype=dtype) < probability


def _sample_edge_disturbance(
    edge_units,
    node_evidence,
    edge_drop_ratio,
    budget_factor,
    enabled,
    dtype,
):
    num_edges = int(edge_units.size(1))
    if num_edges == 0:
        return torch.zeros(0, dtype=torch.bool, device=node_evidence.device)

    source, target = edge_units
    edge_evidence = torch.maximum(node_evidence[source], node_evidence[target])
    disturbance_weight = 1.0 - edge_evidence if enabled else torch.ones_like(edge_evidence)

    if not torch.is_tensor(budget_factor):
        budget_factor = torch.tensor(
            float(budget_factor), device=node_evidence.device, dtype=dtype
        )
    else:
        budget_factor = budget_factor.to(device=node_evidence.device, dtype=dtype)

    probability = (float(edge_drop_ratio) * budget_factor * disturbance_weight).clamp(
        0.0, 1.0
    )
    return torch.rand_like(probability) < probability


def _get_edge_budget_factor(edge_units, node_to_graph, graph_budget_factor):
    if edge_units.size(1) == 0:
        return graph_budget_factor.new_empty((0,))
    return graph_budget_factor[node_to_graph[edge_units[0]]]


def _get_batch_assignment(data, num_nodes, device):
    node_to_graph = getattr(data, "batch", None)
    if node_to_graph is None:
        return torch.zeros(num_nodes, dtype=torch.long, device=device), 1

    node_to_graph = node_to_graph.to(device=device, dtype=torch.long)
    if node_to_graph.dim() != 1 or node_to_graph.numel() != num_nodes:
        raise ValueError("data.batch must contain one graph index per node.")

    num_graphs = int(getattr(data, "num_graphs", 0))
    if num_graphs <= 0:
        num_graphs = int(node_to_graph.max().item()) + 1
    return node_to_graph, num_graphs


def _prepare_graph_budget(value, num_graphs, device, dtype):
    if torch.is_tensor(value):
        budget = value.to(device=device, dtype=dtype).reshape(-1)
    else:
        budget = torch.tensor([float(value)], device=device, dtype=dtype)

    if budget.numel() == 1:
        budget = budget.expand(num_graphs)
    elif budget.numel() != num_graphs:
        raise ValueError("evidence_budget must be scalar or contain one value per graph.")
    return _sanitize_evidence(budget.detach())


def _find_edge_attribute_keys(data, num_edges):
    keys = data.keys
    if callable(keys):
        keys = keys()

    known_edge_keys = {"edge_attr", "edge_weight", "edge_gt_att", "edge_label", "edge_type"}
    result = []
    for key in list(keys):
        if key == "edge_index":
            continue
        value = data[key]
        if not torch.is_tensor(value) or value.dim() == 0 or value.size(0) != num_edges:
            continue

        is_edge_attribute = key in known_edge_keys
        if hasattr(data, "is_edge_attr"):
            try:
                is_edge_attribute = is_edge_attribute or bool(data.is_edge_attr(key))
            except (KeyError, TypeError, ValueError):
                pass
        if is_edge_attribute:
            result.append(key)
    return result


def _validate_node_evidence(node_evidence, num_nodes, device, dtype):
    node_evidence = torch.as_tensor(node_evidence, dtype=dtype, device=device)
    if node_evidence.dim() != 1 or node_evidence.numel() != num_nodes:
        raise ValueError(f"node_evidence must have shape [{num_nodes}].")
    return _sanitize_evidence(node_evidence.detach())


def _sanitize_evidence(value):
    return torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _validate_ratio(name, value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1], got {value}.")
