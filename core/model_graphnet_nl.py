import torch
from torch.nn import Sequential, Linear, ReLU, LayerNorm, LazyLinear, Conv1d
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from .normalization import Normalizer
import os
import torch.nn.functional as F
import torch_scatter
import torch
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import time

class Swish(torch.nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)

class GraphNetBlock(torch.nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, latent_size, in_size1, in_size2):
        super().__init__()
        self._latent_size = latent_size
        
        # update_mesh_edge_net: e_m ij' = f1(xi, xj, e_m ij)
        self.mesh_edge_net = Sequential(Linear(in_size1,self._latent_size),
                                   ReLU(),
                                   Linear(self._latent_size,self._latent_size),
                                   ReLU(),
                                   LayerNorm(self._latent_size))
        
        # update_node_features net (MLP): xi' = f2(xi, sum(e_m ij'))
        self.node_feature_net = Sequential(Linear(in_size2,self._latent_size),
                                   ReLU(),
                                   Linear(self._latent_size,self._latent_size),
                                   ReLU(),
                                   LayerNorm(self._latent_size))    

    def forward(self, graph, mask=None):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        senders = graph.senders
        receivers = graph.receivers     
        node_latents = graph.node_latents
        mesh_edge_latents = graph.mesh_edge_latents
        new_mesh_edge_latents = self.mesh_edge_net(torch.cat([node_latents[senders], node_latents[receivers], mesh_edge_latents], dim=-1))
        aggr = torch_scatter.scatter_add(new_mesh_edge_latents.float(), receivers, dim=0, dim_size=node_latents.shape[0])
        new_node_latents = self.node_feature_net(torch.cat([node_latents, aggr], dim=-1))
        # apply node function

        # add residual connections
        new_node_latents += node_latents
        new_mesh_edge_latents += mesh_edge_latents

        return Data(senders = senders,
                    receivers = receivers, 
                    node_latents = new_node_latents, 
                    mesh_edge_latents = new_mesh_edge_latents,)  
    
class EncodeProcessDecode(torch.nn.Module):
    """Encode-Process-Decode GraphNet model."""

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
        super(EncodeProcessDecode, self).__init__()   
        self._node_feature_size = node_feature_size   
        self._mesh_edge_feature_size = mesh_edge_feature_size
        self._latent_size = latent_size
        self._output_size = output_size      
        self._message_passing_steps = message_passing_steps  
        self._time_window = time_window
        self._timestep = timestep     
        self._output_normalizer = Normalizer(size=output_size, name='output_normalizer')
        self._node_features_normalizer = Normalizer(size = node_feature_size, name='node_features_normalizer')
        self._mesh_edge_normalizer = Normalizer(size = mesh_edge_feature_size, name='mesh_edge_normalizer')
        self._device = device
        # Encoding net (MLP) for node_features
        self.node_encode_net = Sequential(Linear(self._node_feature_size, self._latent_size),
                         ReLU(),
                         Linear(self._latent_size,self._latent_size),
                         ReLU(),
                         LayerNorm(self._latent_size))               
               
        # Encoding net (MLP) for edge_features
        self.mesh_edge_encode_net = Sequential(Linear(self._mesh_edge_feature_size, self._latent_size),
                         ReLU(),
                         Linear(self._latent_size,self._latent_size),
                         ReLU(),
                         LayerNorm(self._latent_size))            
        

        self.graphnet_blocks = torch.nn.ModuleList()
        for _ in range(message_passing_steps):
            self.graphnet_blocks.append(GraphNetBlock(self._latent_size, self._latent_size*3, self._latent_size*2))
        # Decoding net (MLP) for node_features (output)
        # ND: "Node features Decoding"
        self.node_decode_net = Sequential(Conv1d(self._latent_size, 8, 1),
                                          Swish(),
                                          Conv1d(8, self._output_size * self._time_window, 1))
        # self.node_decode_net = Sequential(Linear(self._latent_size,self._latent_size),
        #                 ReLU(),
        #                 Linear(self._latent_size,self._output_size))
                        
    
    def encoder(self, graph) :
        node_features = self._build_node_latent_features(graph)
        mesh_edge_features = self._build_mesh_edge_features(graph)    
        
        node_latents = self.node_encode_net(self._node_features_normalizer(node_features))          
        mesh_edge_latents = self.mesh_edge_encode_net(self._mesh_edge_normalizer(mesh_edge_features))  
        
        return Data(senders = graph.senders, 
                    receivers = graph.receivers, 
                    node_latents = node_latents, 
                    mesh_edge_latents = mesh_edge_latents)
    def forward(self, graph):
        """Encodes and processes a graph, and returns node features."""                     
        # Encoding Layer  
        latent_graph = self.encoder(graph)  
        # Process Layer
        for graphnet_block in self.graphnet_blocks:
            latent_graph = graphnet_block(latent_graph)
        """Decodes node features from graph."""   
        # Decoding node features
        dt = torch.arange(1, self._time_window + 1).repeat_interleave(self._output_size).to(self._device)
        node_latents = latent_graph.node_latents.unsqueeze(0)
        node_latents = node_latents.permute(0, 2, 1)
        decoded_nodes = self.node_decode_net(node_latents)
        decoded_nodes = decoded_nodes.permute(0, 2, 1).squeeze(0)
        delta = decoded_nodes * dt
        delta = delta.reshape(-1, self._time_window, self._output_size)
        delta = delta.permute(1, 0, 2)
        return delta
    def get_output_normalizer(self):
        return self._output_normalizer
    def predict(self, graph) :
        self.eval()
        network_output = self.forward(graph)
        output_normalizer = self.get_output_normalizer()
        delta = output_normalizer.inverse(network_output)
        # delta_temperature = network_output
        cur_world_pos = graph.world_pos.unsqueeze(0).expand(self._time_window, -1, -1)
        cur_pvf = graph.pvf.unsqueeze(0).expand(self._time_window, -1, -1)
        dirichlet_nodes = graph.node_type == 1
        left_node = graph.mesh_pos[:, 0] == torch.min(graph.mesh_pos[:, 0])
        bottom_node = graph.mesh_pos[:, 1] == torch.min(graph.mesh_pos[:, 1])
        delta[:, left_node, 0] = 0
        delta[:, bottom_node, 1] = 0
        next_world_pos = cur_world_pos + delta[:, :, :2]
        next_pvf = cur_pvf + delta[:, :, 2:]
        return next_world_pos, next_pvf
    def loss(self, output, graph) :
        world_pos = graph.world_pos                       # (num_nodes,)
        target_world_pos  = graph.target_world_pos                # (num_nodes, time_window)
        disp = target_world_pos - world_pos.unsqueeze(0).expand(self._time_window, -1, -1) # (num_nodes, time_window)

        pvf = graph.pvf                       # (num_nodes,)
        target_pvf = graph.target_pvf                # (num_nodes, time_window)
        delta_pvf = target_pvf - pvf.unsqueeze(0).expand(self._time_window, -1, -1)  # (num_nodes, time_window)

        target_delta = torch.cat([disp, delta_pvf], dim = -1)


        normalizer = self.get_output_normalizer()
        target_normalized = normalizer(target_delta)        # (num_nodes, time_window)

        node_type = graph.node_type                            # (num_nodes,)
        error = (output - target_normalized) ** 2               # (num_nodes,)
        disp_loss = torch.mean(torch.sum(error[:, :, :2], dim = 2), dim = 1)
        pvf_loss = torch.mean(torch.sum(error[:, :, 2:], dim = 2), dim = 1)     # scalar
        window_avg_disp_loss, window_avg_pvf_loss = torch.mean(disp_loss), torch.mean(pvf_loss)
        total_loss = window_avg_disp_loss + window_avg_pvf_loss
        return total_loss, window_avg_disp_loss, window_avg_pvf_loss
    def fem_physical_loss(self, network_output, graph) :
        return # to be developed
    def pde_physical_loss(self, network_output, graph) :
        return # to be developed
    
    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "model_weights.pth"))
        torch.save(self._output_normalizer, os.path.join(path, "output_normalizer.pth"))
        torch.save(self._node_features_normalizer, os.path.join(path, "node_features_normalizer.pth"))
        torch.save(self._mesh_edge_normalizer, os.path.join(path, "mesh_edge_features_normalizer.pth"))
        
    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth"), weights_only=True))
        self._output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self._node_features_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self._mesh_edge_normalizer = torch.load(os.path.join(path, "mesh_edge_features_normalizer.pth"))
    
    def _build_node_latent_features(self, graph) :
        node_type_onehot = F.one_hot(graph.node_type).to(torch.float)
        node_latent_features = torch.cat(
            (graph.pvf, graph.mat_param_D, graph.mat_param_X, node_type_onehot), 
            dim = -1
            )
        return node_latent_features
    def _build_mesh_edge_features(self, graph) :
        senders, receivers = graph.senders, graph.receivers
        relative_mesh_pos = graph.mesh_pos[senders] - graph.mesh_pos[receivers]
        norm_rel_mesh_pos = torch.norm(relative_mesh_pos, dim=-1, keepdim=True)
        relative_world_pos = graph.world_pos[senders] - graph.world_pos[receivers]
        norm_rel_world_pos = torch.norm(relative_mesh_pos, dim=-1, keepdim=True)
        nodal_pvf_gradient = graph.pvf[senders] - graph.pvf[receivers]
        # concatenate the mesh edges data together
        mesh_edge_features = torch.cat(
            (relative_mesh_pos, norm_rel_mesh_pos, relative_world_pos, norm_rel_world_pos, nodal_pvf_gradient), 
            dim = -1
            )
        return mesh_edge_features

