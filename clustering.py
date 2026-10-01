from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

IEC_GOOD_KMEANS_CLUSTERS = 300


@dataclass(frozen=True)
class ClusterState:
    semantic_labels: torch.Tensor
    context_labels: torch.Tensor
    semantic_prototypes: torch.Tensor
    environment_bank: torch.Tensor
    donor_indices: torch.Tensor
    refresh_id: int


def fit_cluster_state(
    graph_ids,
    semantic_representations,
    environment_projections,
    valid_environment,
    num_graphs,
    semantic_clusters,
    context_clusters,
    seed,
    refresh_id,
):


    graph_ids = torch.as_tensor(graph_ids, dtype=torch.long).cpu().view(-1)
    semantic = F.normalize(
        torch.as_tensor(semantic_representations, dtype=torch.float32).cpu(), dim=-1
    )
    environment = F.normalize(
        torch.as_tensor(environment_projections, dtype=torch.float32).cpu(), dim=-1
    )
    valid_environment = torch.as_tensor(valid_environment, dtype=torch.bool).cpu().view(-1)

    if semantic.dim() != 2 or environment.dim() != 2:
        raise ValueError("semantic/environment representations must be rank-2 tensors.")
    if graph_ids.numel() != semantic.size(0) or graph_ids.numel() != environment.size(0):
        raise ValueError("Each graph must have one semantic and one environment representation.")
    if valid_environment.numel() != graph_ids.numel():
        raise ValueError("valid_environment length mismatch.")
    if int(num_graphs) < 1:
        raise ValueError("num_graphs must be positive.")

    _validate_complete_ids(graph_ids, int(num_graphs))

    semantic_bank = torch.zeros((num_graphs, semantic.size(1)), dtype=semantic.dtype)
    environment_bank = torch.zeros((num_graphs, environment.size(1)), dtype=environment.dtype)
    environment_valid_bank = torch.zeros(num_graphs, dtype=torch.bool)
    semantic_bank[graph_ids] = semantic
    environment_bank[graph_ids] = environment
    environment_valid_bank[graph_ids] = valid_environment

    semantic_labels = _kmeans_labels(
        semantic_bank, requested_clusters=int(semantic_clusters), seed=int(seed) + 17 * refresh_id
    )

    context_labels = torch.full((num_graphs,), -1, dtype=torch.long)
    valid_ids = environment_valid_bank.nonzero(as_tuple=False).view(-1)
    if valid_ids.numel() > 0:
        context_labels[valid_ids] = _kmeans_labels(
            environment_bank[valid_ids],
            requested_clusters=int(context_clusters),
            seed=int(seed) + 31 * refresh_id,
        )

    semantic_prototypes = _context_balanced_prototypes(
        semantic_bank, semantic_labels, context_labels
    )
    donor_indices = _build_cross_context_donors(
        context_labels, seed=int(seed) + 47 * refresh_id
    )

    return ClusterState(
        semantic_labels=semantic_labels,
        context_labels=context_labels,
        semantic_prototypes=semantic_prototypes,
        environment_bank=environment_bank,
        donor_indices=donor_indices,
        refresh_id=int(refresh_id),
    )


def _validate_complete_ids(graph_ids, num_graphs):
    if graph_ids.numel() != num_graphs:
        raise ValueError(
            f"Cluster refresh expected {num_graphs} graph representations, "
            f"received {graph_ids.numel()}."
        )
    sorted_ids = torch.sort(graph_ids).values
    expected = torch.arange(num_graphs, dtype=torch.long)
    if not torch.equal(sorted_ids, expected):
        raise ValueError("pretrain_id must be a complete permutation of [0, num_graphs).")


def _kmeans_labels(representations, requested_clusters, seed):
    count = int(representations.size(0))
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    clusters = max(1, min(int(requested_clusters), count))
    if clusters == 1:
        return torch.zeros(count, dtype=torch.long)

    model = KMeans(n_clusters=clusters, random_state=int(seed), n_init=10)
    labels = model.fit_predict(representations.numpy())
    return torch.from_numpy(np.asarray(labels, dtype=np.int64))


def _context_balanced_prototypes(semantic_bank, semantic_labels, context_labels):
    num_semantic = int(semantic_labels.max().item()) + 1
    prototypes = []

    for semantic_id in range(num_semantic):
        semantic_mask = semantic_labels == semantic_id
        available_contexts = torch.unique(context_labels[semantic_mask])
        available_contexts = available_contexts[available_contexts >= 0]

        context_means = []
        for context_id in available_contexts.tolist():
            mask = semantic_mask & (context_labels == int(context_id))
            if bool(mask.any()):
                context_means.append(semantic_bank[mask].mean(dim=0))

        if context_means:
            prototype = torch.stack(context_means, dim=0).mean(dim=0)
        else:
            prototype = semantic_bank[semantic_mask].mean(dim=0)

        prototypes.append(F.normalize(prototype, dim=0))

    return torch.stack(prototypes, dim=0)


def _build_cross_context_donors(context_labels, seed):
    donor = torch.full_like(context_labels, -1)
    valid_ids = (context_labels >= 0).nonzero(as_tuple=False).view(-1)
    generator = torch.Generator().manual_seed(int(seed))

    for graph_id in valid_ids.tolist():
        candidates = valid_ids[context_labels[valid_ids] != context_labels[graph_id]]
        if candidates.numel() == 0:
            continue
        position = int(torch.randint(candidates.numel(), (1,), generator=generator).item())
        donor[graph_id] = candidates[position]

    return donor
