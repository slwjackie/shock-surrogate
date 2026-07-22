"""Dependency-light conservative MeshGraphNet-style surrogate."""

from __future__ import annotations

import torch
import torch.nn as nn

from hypersonic.positivity import positivity_preserving_blend


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int = 2) -> nn.Sequential:
    blocks: list[nn.Module] = []
    d = in_dim
    for _ in range(max(1, layers - 1)):
        blocks.extend((nn.Linear(d, hidden), nn.SiLU()))
        d = hidden
    blocks.append(nn.Linear(d, out_dim))
    return nn.Sequential(*blocks)


class MessagePassingBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = _mlp(3 * hidden, hidden, hidden)
        self.node_mlp = _mlp(2 * hidden, hidden, hidden)
        self.edge_norm = nn.LayerNorm(hidden)
        self.node_norm = nn.LayerNorm(hidden)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor):
        src, dst = edge_index[0].long(), edge_index[1].long()
        message = self.edge_mlp(torch.cat((nodes[src], nodes[dst], edges), dim=-1))
        edges = self.edge_norm(edges + message)
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, dst, edges)
        counts = torch.zeros(nodes.shape[0], 1, device=nodes.device, dtype=nodes.dtype)
        counts.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=nodes.device, dtype=nodes.dtype))
        aggregate = aggregate / counts.clamp_min(1.0)
        nodes = self.node_norm(nodes + self.node_mlp(torch.cat((nodes, aggregate), dim=-1)))
        return nodes, edges


class ConservativeMeshGNN(nn.Module):
    """Graph surrogate for unstructured 2-D or 3-D compressible meshes."""

    def __init__(self, state_channels: int = 4, coordinate_dim: int = 2, hidden: int = 128, message_passing_steps: int = 8, gamma: float = 1.4, rho_floor: float = 1e-6, p_floor: float = 1e-6):
        super().__init__()
        self.state_channels = int(state_channels)
        self.coordinate_dim = int(coordinate_dim)
        self.gamma = float(gamma)
        self.rho_floor = float(rho_floor)
        self.p_floor = float(p_floor)
        self.node_encoder = _mlp(state_channels + coordinate_dim, hidden, hidden)
        self.edge_encoder = _mlp(coordinate_dim + 1, hidden, hidden)
        self.processor = nn.ModuleList([MessagePassingBlock(hidden) for _ in range(message_passing_steps)])
        self.decoder = _mlp(hidden, state_channels, hidden)

    @staticmethod
    def _graph_mean(values: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        total = torch.zeros(n_graphs, values.shape[1], device=values.device, dtype=values.dtype)
        total.index_add_(0, batch, values)
        counts = torch.zeros(n_graphs, 1, device=values.device, dtype=values.dtype)
        counts.index_add_(0, batch, torch.ones(batch.shape[0], 1, device=values.device, dtype=values.dtype))
        return total / counts.clamp_min(1.0)

    def forward(self, state: torch.Tensor, positions: torch.Tensor, edge_index: torch.Tensor, dt: float = 1.0, batch: torch.Tensor | None = None):
        if state.ndim != 2 or positions.ndim != 2:
            raise ValueError("state and positions must be node-major matrices")
        if positions.shape[1] != self.coordinate_dim:
            raise ValueError("position dimension does not match coordinate_dim")
        if batch is None:
            batch = torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
        n_graphs = int(batch.max().item()) + 1
        src, dst = edge_index[0].long(), edge_index[1].long()
        rel = positions[dst] - positions[src]
        edge_length = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        nodes = self.node_encoder(torch.cat((state, positions), dim=-1))
        edges = self.edge_encoder(torch.cat((rel, edge_length), dim=-1))
        for block in self.processor:
            nodes, edges = block(nodes, edges, edge_index)
        increment = self.decoder(nodes)
        graph_mean = self._graph_mean(increment, batch, n_graphs)
        increment = increment - graph_mean[batch]
        candidate = state + float(dt) * increment
        safe = candidate.clone()
        thetas = []
        for graph_id in range(n_graphs):
            mask = batch == graph_id
            ref_g = state[mask].T.unsqueeze(0)
            cand_g = candidate[mask].T.unsqueeze(0)
            safe_g, theta = positivity_preserving_blend(ref_g, cand_g, gamma=self.gamma, rho_floor=self.rho_floor, p_floor=self.p_floor)
            safe[mask] = safe_g.squeeze(0).T
            thetas.append(theta.squeeze(0))
        return safe, {"increment": increment, "positivity_theta": torch.stack(thetas)}
