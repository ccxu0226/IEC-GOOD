from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class BatchedRegionProposal:
    hard_masks: torch.Tensor
    st_masks: torch.Tensor
    valid_regions: torch.Tensor
    region_sizes: torch.Tensor
    selection_log_probs: torch.Tensor


class ConnectedRegionGenerator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_regions,
        min_region_size,
        max_region_coverage,
        temperature,
        validate_proposals=False,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_regions <= 0:
            raise ValueError(f"num_regions must be positive, got {num_regions}.")
        if min_region_size <= 0:
            raise ValueError(f"min_region_size must be positive, got {min_region_size}.")
        if not 0.0 < max_region_coverage < 1.0:
            raise ValueError(f"max_region_coverage must lie in (0, 1), got {max_region_coverage}.")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_regions = int(num_regions)
        self.min_region_size = int(min_region_size)
        self.max_region_coverage = float(max_region_coverage)
        self.temperature = float(temperature)
        self.validate_proposals = bool(validate_proposals)

        self.root_scorer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.frontier_scorer = nn.Sequential(
            nn.Linear(self.input_dim * 3, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, node_embeddings, edge_index, batch, num_graphs):
        self._validate_inputs(node_embeddings, edge_index, batch, num_graphs)

        num_graphs = int(num_graphs)
        num_nodes = int(node_embeddings.size(0))
        device = node_embeddings.device
        dtype = node_embeddings.dtype

        graph_node_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        graph_node_counts.index_add_(0, batch, torch.ones(num_nodes, dtype=torch.long, device=device))

        coverage_budget = torch.floor(graph_node_counts.to(dtype) * self.max_region_coverage).long()
        nonempty_complement_budget = (graph_node_counts - 1).clamp_min(0)
        maximum_selected_nodes = torch.minimum(coverage_budget, nonempty_complement_budget)
        maximum_region_count = torch.div(
            maximum_selected_nodes,
            self.min_region_size,
            rounding_mode="floor",
        ).clamp(max=self.num_regions)

        source = torch.cat([edge_index[0], edge_index[1]], dim=0)
        target = torch.cat([edge_index[1], edge_index[0]], dim=0)
        non_self = source != target
        source = source[non_self]
        target = target[non_self]

        available = torch.ones(num_nodes, dtype=torch.bool, device=device)
        root_logits = self.root_scorer(node_embeddings).squeeze(-1)

        hard_slots = []
        st_slots = []
        valid_slots = []
        size_slots = []
        log_prob_slots = []

        for slot_index in range(self.num_regions):
            slot_allowed_by_graph = slot_index < maximum_region_count
            available_counts = self._segment_sum(available.to(dtype), batch, num_graphs)
            enough_nodes = available_counts >= float(self.min_region_size)
            root_candidates = available & slot_allowed_by_graph[batch] & enough_nodes[batch]

            if self.min_region_size > 1:
                root_candidates = root_candidates & self._has_available_neighbor(
                    available=available,
                    source=source,
                    target=target,
                    num_nodes=num_nodes,
                    dtype=dtype,
                )

            root_hard, root_st, root_valid, root_log_prob = self._batched_st_sample(
                logits=root_logits,
                candidate_mask=root_candidates,
                batch=batch,
                num_graphs=num_graphs,
            )

            region_hard = root_hard.bool()
            region_st = root_st
            region_valid = root_valid
            selection_log_probability = root_log_prob
            root_context = self._segment_sum(root_st.unsqueeze(-1) * node_embeddings, batch, num_graphs)

            for _ in range(1, self.min_region_size):
                frontier = self._frontier_mask(
                    region_hard=region_hard,
                    available=available,
                    source=source,
                    target=target,
                    num_nodes=num_nodes,
                    dtype=dtype,
                )
                frontier = frontier & region_valid[batch]

                region_mass = self._segment_sum(region_st, batch, num_graphs).clamp_min(1.0)
                region_context = self._segment_sum(
                    region_st.unsqueeze(-1) * node_embeddings,
                    batch,
                    num_graphs,
                ) / region_mass.unsqueeze(-1)

                scorer_input = torch.cat(
                    [node_embeddings, region_context[batch], root_context[batch]],
                    dim=-1,
                )
                frontier_logits = self.frontier_scorer(scorer_input).squeeze(-1)

                node_hard, node_st, node_valid, node_log_prob = self._batched_st_sample(
                    logits=frontier_logits,
                    candidate_mask=frontier,
                    batch=batch,
                    num_graphs=num_graphs,
                )

                region_valid = region_valid & node_valid
                region_hard = region_hard | node_hard.bool()
                region_st = region_st + node_st
                selection_log_probability = selection_log_probability + node_log_prob

            valid_by_node = region_valid[batch]
            final_hard = region_hard & valid_by_node
            final_st = region_st * valid_by_node.to(dtype)
            final_sizes = self._segment_sum(final_hard.to(dtype), batch, num_graphs).long()
            final_log_probability = selection_log_probability * region_valid.to(dtype)
            available = available & ~final_hard

            hard_slots.append(final_hard)
            st_slots.append(final_st)
            valid_slots.append(region_valid)
            size_slots.append(final_sizes)
            log_prob_slots.append(final_log_probability)

        proposal = BatchedRegionProposal(
            hard_masks=torch.stack(hard_slots, dim=0),
            st_masks=torch.stack(st_slots, dim=0),
            valid_regions=torch.stack(valid_slots, dim=0),
            region_sizes=torch.stack(size_slots, dim=0),
            selection_log_probs=torch.stack(log_prob_slots, dim=0),
        )

        if self.validate_proposals:
            self._validate_batched_proposal(
                proposal=proposal,
                batch=batch,
                maximum_selected_nodes=maximum_selected_nodes,
                num_graphs=num_graphs,
            )

        return proposal

    def _batched_st_sample(self, logits, candidate_mask, batch, num_graphs):
        dtype = logits.dtype
        device = logits.device
        num_nodes = int(logits.numel())

        if self.training:
            uniform_noise = torch.rand_like(logits).clamp_(1e-8, 1.0 - 1e-8)
            gumbel_noise = -torch.log(-torch.log(uniform_noise))
            sampling_scores = (logits + gumbel_noise) / self.temperature
        else:
            sampling_scores = logits / self.temperature

        masked_scores = torch.where(
            candidate_mask,
            sampling_scores,
            torch.full_like(sampling_scores, -torch.inf),
        )

        graph_max = torch.full((num_graphs,), -torch.inf, dtype=dtype, device=device)
        graph_max.scatter_reduce_(
            0,
            batch,
            masked_scores,
            reduce="amax",
            include_self=True,
        )

        candidate_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        candidate_counts.index_add_(0, batch, candidate_mask.long())
        graph_valid = candidate_counts > 0
        safe_graph_max = torch.where(graph_valid, graph_max, torch.zeros_like(graph_max))

        exponentials = torch.where(
            candidate_mask,
            torch.exp(masked_scores - safe_graph_max[batch]),
            torch.zeros_like(masked_scores),
        )

        graph_denominator = torch.zeros(num_graphs, dtype=dtype, device=device)
        graph_denominator.index_add_(0, batch, exponentials)
        soft_choice = exponentials / graph_denominator[batch].clamp_min(1e-12)

        global_indices = torch.arange(num_nodes, device=device)
        is_group_maximum = candidate_mask & (masked_scores == graph_max[batch])
        selected_candidates = torch.where(
            is_group_maximum,
            global_indices,
            torch.full_like(global_indices, num_nodes),
        )

        selected_indices = torch.full(
            (num_graphs,),
            num_nodes,
            dtype=torch.long,
            device=device,
        )
        selected_indices.scatter_reduce_(
            0,
            batch,
            selected_candidates,
            reduce="amin",
            include_self=True,
        )

        safe_selected_indices = selected_indices.clamp(max=num_nodes - 1)
        hard_choice = torch.zeros(num_nodes, dtype=dtype, device=device)
        hard_choice.scatter_add_(0, safe_selected_indices, graph_valid.to(dtype))
        st_choice = hard_choice + soft_choice - soft_choice.detach()

        selected_probability = soft_choice.index_select(0, safe_selected_indices).clamp_min(1e-12)
        log_probability = torch.where(
            graph_valid,
            selected_probability.log(),
            torch.zeros_like(selected_probability),
        )

        return hard_choice, st_choice, graph_valid, log_probability

    @staticmethod
    def _segment_sum(values, batch, num_graphs):
        output = values.new_zeros((num_graphs,) + tuple(values.shape[1:]))
        output.index_add_(0, batch, values)
        return output

    @staticmethod
    def _has_available_neighbor(available, source, target, num_nodes, dtype):
        active_edges = available[source] & available[target]
        neighbor_votes = torch.zeros(num_nodes, dtype=dtype, device=available.device)
        neighbor_votes.index_add_(0, source, active_edges.to(dtype))
        return neighbor_votes > 0

    @staticmethod
    def _frontier_mask(region_hard, available, source, target, num_nodes, dtype):
        active_edges = region_hard[source] & available[target] & ~region_hard[target]
        frontier_votes = torch.zeros(num_nodes, dtype=dtype, device=available.device)
        frontier_votes.index_add_(0, target, active_edges.to(dtype))
        return frontier_votes > 0

    def _validate_inputs(self, node_embeddings, edge_index, batch, num_graphs):
        if node_embeddings.dim() != 2:
            raise ValueError(f"node_embeddings must have shape [N, D], got {node_embeddings.shape}.")
        if node_embeddings.size(1) != self.input_dim:
            raise ValueError(
                f"Node embedding dimension mismatch: expected {self.input_dim}, got {node_embeddings.size(1)}."
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}.")
        if edge_index.dtype != torch.long:
            raise ValueError(f"edge_index must use torch.long indices, got {edge_index.dtype}.")
        if batch.dim() != 1:
            raise ValueError(f"batch must have shape [N], got {batch.shape}.")
        if batch.numel() != node_embeddings.size(0):
            raise ValueError("batch length does not match the number of nodes.")
        if node_embeddings.size(0) == 0:
            raise ValueError("Cannot generate regions for an empty mini-batch.")
        if int(num_graphs) <= 0:
            raise ValueError(f"num_graphs must be positive, got {num_graphs}.")
        if edge_index.device != node_embeddings.device or batch.device != node_embeddings.device:
            raise ValueError("node_embeddings, edge_index and batch must share one device.")

    @torch.no_grad()
    def _validate_batched_proposal(self, proposal, batch, maximum_selected_nodes, num_graphs):
        memberships = proposal.hard_masks.sum(dim=0)

        if bool((memberships > 1).any()):
            raise RuntimeError("Generated regions overlap.")

        selected_per_graph = self._segment_sum(
            memberships.to(torch.long),
            batch,
            num_graphs,
        )

        if bool((selected_per_graph > maximum_selected_nodes).any()):
            raise RuntimeError("Generated regions exceed the coverage budget.")

        valid_sizes = proposal.region_sizes[proposal.valid_regions]

        if valid_sizes.numel() > 0 and bool((valid_sizes != self.min_region_size).any()):
            raise RuntimeError("A valid region has an unexpected size.")
