import os
import time
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LayerNorm, Conv1d
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import torch_scatter

from .normalization import Normalizer


class Swish(torch.nn.Module):
    """Swish activation function."""
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GraphNetBlock(torch.nn.Module):
    """Graph Network block with residual connections."""
    def __init__(self, latent_size, node_edge_input_size, node_input_size):
        super().__init__()
        self.latent_size = latent_size

        self.mesh_edge_net = Sequential(
            Linear(node_edge_input_size, latent_size),
            ReLU(),
            Linear(latent_size, latent_size),
            ReLU(),
            LayerNorm(latent_size)
        )

        self.node_feature_net = Sequential(
            Linear(node_input_size, latent_size),
            ReLU(),
            Linear(latent_size, latent_size),
            ReLU(),
            LayerNorm(latent_size)
        )

    def forward(self, graph, mask=None):
        senders, receivers = graph.senders, graph.receivers
        node_latents, mesh_edge_latents = graph.node_latents, graph.mesh_edge_latents

        edge_input = torch.cat([node_latents[senders], node_latents[receivers], mesh_edge_latents], dim=-1)
        new_mesh_edge_latents = self.mesh_edge_net(edge_input)

        aggr = torch_scatter.scatter_add(new_mesh_edge_latents.float(), receivers, dim=0, dim_size=node_latents.size(0))
        node_input = torch.cat([node_latents, aggr], dim=-1)
        new_node_latents = self.node_feature_net(node_input)

        return Data(
            senders=senders,
            receivers=receivers,
            node_latents=new_node_latents + node_latents,
            mesh_edge_latents=new_mesh_edge_latents + mesh_edge_latents,
        )


class EncodeProcessDecode(torch.nn.Module):
    """Encode-Process-Decode architecture for GNN."""

    def __init__(self,
                 node_feature_size,
                 mesh_edge_feature_size,
                 output_size,
                 latent_size,
                 timestep,
                 time_window,
                 message_passing_steps,
                 device,
                 name='EncodeProcessDecode'):
        super().__init__()
        self.name = name
        self.device = device

        self.node_feature_size = node_feature_size
        self.mesh_edge_feature_size = mesh_edge_feature_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.time_window = time_window
        self.timestep = timestep
        self.message_passing_steps = message_passing_steps

        # Normalizers
        self.output_normalizer = Normalizer(output_size, 'output_normalizer')
        self.node_normalizer = Normalizer(node_feature_size, 'node_features_normalizer')
        self.edge_normalizer = Normalizer(mesh_edge_feature_size, 'mesh_edge_normalizer')

        # Encoders
        self.node_encoder = self._make_encoder(node_feature_size)
        self.edge_encoder = self._make_encoder(mesh_edge_feature_size)

        # GNN core
        self.graphnet_blocks = torch.nn.ModuleList([
            GraphNetBlock(latent_size, latent_size * 3, latent_size * 2)
            for _ in range(message_passing_steps)
        ])

        # Decoder
        self.node_decoder = Sequential(
            Conv1d(latent_size, 8, 1),
            Swish(),
            Conv1d(8, output_size * time_window, 1)
        )

    def _make_encoder(self, input_size):
        return Sequential(
            Linear(input_size, self.latent_size),
            ReLU(),
            Linear(self.latent_size, self.latent_size),
            ReLU(),
            LayerNorm(self.latent_size)
        )

    def forward(self, graph):
        latent_graph = self._encode(graph)

        for block in self.graphnet_blocks:
            latent_graph = block(latent_graph)

        node_latents = latent_graph.node_latents.unsqueeze(0).permute(0, 2, 1)
        decoded = self.node_decoder(node_latents).permute(0, 2, 1).squeeze(0)

        dt = torch.arange(1, self.time_window + 1).repeat_interleave(self.output_size).to(self.device)
        delta = (decoded * dt).reshape(-1, self.time_window, self.output_size).permute(1, 0, 2)

        return delta

    def predict(self, graph):
        self.eval()
        output = self.forward(graph)
        delta = self.output_normalizer.inverse(output)

        pos = graph.world_pos.unsqueeze(0).expand(self.time_window, -1, -1)
        pvf = graph.pvf.unsqueeze(0).expand(self.time_window, -1, -1)

        left = graph.mesh_pos[:, 0] == torch.min(graph.mesh_pos[:, 0])
        bottom = graph.mesh_pos[:, 1] == torch.min(graph.mesh_pos[:, 1])

        delta[:, left, 0] = 0
        delta[:, bottom, 1] = 0

        next_pos = pos + delta[:, :, :2]
        next_pvf = pvf + delta[:, :, 2:]

        return next_pos, next_pvf

    def loss(self, output, graph):
        world_pos = graph.world_pos
        target_world_pos = graph.target_world_pos
        disp = target_world_pos - world_pos.unsqueeze(0)

        pvf = graph.pvf
        target_pvf = graph.target_pvf
        delta_pvf = target_pvf - pvf.unsqueeze(0)

        target_delta = torch.cat([disp, delta_pvf], dim=-1)
        normalized_target = self.output_normalizer(target_delta)

        error = (output - normalized_target) ** 2
        disp_loss = torch.mean(torch.sum(error[:, :, :2], dim=2), dim=1)
        pvf_loss = torch.mean(torch.sum(error[:, :, 2:], dim=2), dim=1)

        return torch.mean(disp_loss + pvf_loss), torch.mean(disp_loss), torch.mean(pvf_loss)

    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "model_weights.pth"))
        torch.save(self.output_normalizer, os.path.join(path, "output_normalizer.pth"))
        torch.save(self.node_normalizer, os.path.join(path, "node_features_normalizer.pth"))
        torch.save(self.edge_normalizer, os.path.join(path, "mesh_edge_features_normalizer.pth"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth"), weights_only=True))
        self.output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self.node_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self.edge_normalizer = torch.load(os.path.join(path, "mesh_edge_features_normalizer.pth"))

    def _encode(self, graph):
        node_feats = self._build_node_features(graph)
        edge_feats = self._build_edge_features(graph)

        return Data(
            senders=graph.senders,
            receivers=graph.receivers,
            node_latents=self.node_encoder(self.node_normalizer(node_feats)),
            mesh_edge_latents=self.edge_encoder(self.edge_normalizer(edge_feats))
        )

    def _build_node_features(self, graph):
        one_hot_type = F.one_hot(graph.node_type).float()
        return torch.cat([graph.pvf, graph.mat_param_D, graph.mat_param_X, one_hot_type], dim=-1)

    def _build_edge_features(self, graph):
        s, r = graph.senders, graph.receivers
        rel_mesh = graph.mesh_pos[s] - graph.mesh_pos[r]
        dist_mesh = torch.norm(rel_mesh, dim=-1, keepdim=True)
        rel_world = graph.world_pos[s] - graph.world_pos[r]
        dist_world = torch.norm(rel_world, dim=-1, keepdim=True)
        pvf_grad = graph.pvf[s] - graph.pvf[r]

        return torch.cat([rel_mesh, dist_mesh, rel_world, dist_world, pvf_grad], dim=-1)
