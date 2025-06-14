import json
import torch
from torch_geometric.data import Data, Dataset
import os
from .utils import *
import numpy as np
class HydrogelDataset(Dataset):
    def __init__(self, data_dir, add_targets, split_frames, add_noise, time_window):
        """
        Generates synthetic dataset for material deformation use case.

        Args:
            num_graphs (int): Number of graphs in the dataset.
            num_nodes (int): Number of nodes per graph.
            num_features (int): Number of features per node.
            num_material_params (int): Number of material parameters.
        """
        super(HydrogelDataset, self).__init__()
        self.data_dir = data_dir
        self.add_targets = add_targets
        self.add_noise = add_noise
        self.time_window = time_window
        self.split_frames = split_frames
        self.file_name_list = [filename for filename in sorted(os.listdir(data_dir)) if not os.path.isdir(os.path.join(data_dir, filename))]
    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        # Randomly generate node features
        file_name = self.file_name_list[idx]
        data = np.load(os.path.join(self.data_dir, file_name))

        decomposed_connectivity = triangles_to_edges(torch.tensor(data['node_connectivity'], dtype = torch.int))['two_way_connectivity']
        displacement = torch.tensor(data["displacement_list"], dtype=torch.float)
        chem_pot = torch.tensor(data["chem_pot_list"], dtype=torch.float).unsqueeze(-1)
        mesh_pos = torch.tensor(data["mesh_pos"], dtype=torch.float)
        mat_param_lambda = torch.tensor(data["mat_param_lambda"], dtype=torch.float).unsqueeze(-1)
        mat_param_A = torch.tensor(data["mat_param_A"], dtype=torch.float).unsqueeze(-1)
        cells = torch.tensor(data['node_connectivity'])
        node_type = torch.tensor(data["node_type"], dtype = torch.long)
        # edge_index = torch.cat((decomposed_connectivity[0].reshape(1, -1), decomposed_connectivity[1].reshape(1, -1)), dim=0)
        senders, receivers = decomposed_connectivity[0], decomposed_connectivity[1]
        if self.add_targets :
            target_displacement = torch.stack([displacement[i + 1 : i + 1 + self.time_window] for i in range(len(displacement) - self.time_window)], dim=0)
            target_chem_pot = torch.stack([chem_pot[i + 1 : i + 1 + self.time_window] for i in range(len(chem_pot) - self.time_window)], dim=0)
        if self.split_frames & self.add_targets :
            #list of data (frame)
            frames = []
            for idx in range(target_displacement.shape[0]) :
                displacement_t = displacement[idx]
                target_displacement_t = target_displacement[idx]
                chem_pot_t = chem_pot[idx]
                target_chem_pot_t = target_chem_pot[idx]
                if self.add_noise :
                    displacement_noise_scale = (torch.max(displacement) - torch.min(displacement)) * 0.01
                    chem_pot_noise_scale = (torch.max(chem_pot) - torch.min(chem_pot)) * 0.01
                    displacement_noise = torch.zeros_like(displacement_t) + displacement_noise_scale * torch.randn_like(displacement_t)
                    chem_pot_noise = torch.zeros_like(chem_pot_t) + chem_pot_noise_scale * torch.randn_like(chem_pot_t)
                    displacement_t += displacement_noise
                    chem_pot_t += chem_pot_noise
                frame = Data(displacement = displacement_t, 
                             target_displacement = target_displacement_t, 
                             chem_pot = chem_pot_t,
                             target_chem_pot = target_chem_pot_t,
                             mesh_pos = mesh_pos,  
                             senders = senders, 
                             receivers = receivers, 
                             cells = cells,
                             node_type = node_type,
                             mat_param_lambda = mat_param_lambda,
                             mat_param_A = mat_param_A)
                frames.append(frame)
            return frames


        return Data(displacement = displacement, 
                    chem_pot = chem_pot,
                    mesh_pos = mesh_pos,  
                    senders = senders, 
                    receivers = receivers, 
                    cells = cells,
                    node_type = node_type)
    def get_name(self, idx) :
        return self.file_name_list[idx]
    

class HydrogelNonLinearDataset(Dataset):
    def __init__(
        self,
        data_dir,
        add_targets=True,
        split_frames=True,
        add_noise=True,
        time_window=1,
        target_config=None  # {"world_pos": {"noise": 0.01}, "pvf": {"noise": 0.02}}
    ):
        super().__init__()
        self.data_dir = data_dir
        self.add_targets = add_targets
        self.split_frames = split_frames
        self.add_noise = add_noise
        self.time_window = time_window
        self.target_config = target_config or {}
        self.file_name_list = sorted([
            f for f in os.listdir(data_dir) if not os.path.isdir(os.path.join(data_dir, f))
        ])

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        file_name = self.file_name_list[idx]
        data = np.load(os.path.join(self.data_dir, file_name))

        world_pos = torch.tensor(data["world_pos"], dtype=torch.float)
        pvf = torch.tensor(data["pvf"], dtype=torch.float).unsqueeze(-1)
        mesh_pos = torch.tensor(data["mesh_pos"], dtype=torch.float)
        node_type = torch.tensor(data["node_type"], dtype=torch.long).squeeze(-1)
        cells = torch.tensor(data["node_connectivity"], dtype=torch.long)
        senders, receivers = triangles_to_edges(cells)["two_way_connectivity"]

        mat_param_D = torch.full((node_type.shape[0], 1), float(data["mat_param_D"]))
        mat_param_X = torch.full((node_type.shape[0], 1), float(data["mat_param_X"]))

        # Targets
        targets = {}
        if self.add_targets:
            if "world_pos" in self.target_config:
                targets["target_world_pos"] = torch.stack([
                    world_pos[i + 1: i + 1 + self.time_window]
                    for i in range(len(world_pos) - self.time_window)
                ])
            if "pvf" in self.target_config:
                targets["target_pvf"] = torch.stack([
                    pvf[i + 1: i + 1 + self.time_window]
                    for i in range(len(pvf) - self.time_window)
                ])

        if self.split_frames and self.add_targets:
            frames = []
            num_frames = len(world_pos) - self.time_window
            for t in range(num_frames):
                frame_data = {
                    "world_pos": world_pos[t],
                    "pvf": pvf[t],
                    "mesh_pos": mesh_pos,
                    "senders": senders,
                    "receivers": receivers,
                    "cells": cells,
                    "node_type": node_type,
                    "mat_param_D": mat_param_D,
                    "mat_param_X": mat_param_X,
                }

                if "world_pos" in self.target_config:
                    frame_data["target_world_pos"] = targets["target_world_pos"][t]
                if "pvf" in self.target_config:
                    frame_data["target_pvf"] = targets["target_pvf"][t]

                if self.add_noise:
                    if "world_pos" in self.target_config:
                        noise = self.target_config["world_pos"].get("noise", 0.0)
                        if noise > 0:
                            edge_lengths = (mesh_pos[senders] - mesh_pos[receivers]).norm(dim=-1)
                            avg_edge_length = edge_lengths.mean()
                            frame_data["world_pos"] += avg_edge_length * noise * torch.randn_like(frame_data["world_pos"])

                    if "pvf" in self.target_config:
                        noise = self.target_config["pvf"].get("noise", 0.0)
                        if noise > 0:
                            frame_data["pvf"] += pvf.std() * noise * torch.randn_like(frame_data["pvf"])

                frames.append(Data(**frame_data))
            return frames

        # If not splitting into frames
        full_data = {
            "world_pos": world_pos,
            "pvf": pvf,
            "mesh_pos": mesh_pos,
            "senders": senders,
            "receivers": receivers,
            "cells": cells,
            "node_type": node_type,
            "mat_param_D": mat_param_D,
            "mat_param_X": mat_param_X,
        }
        return Data(**full_data)

    def get_name(self, idx):
        return self.file_name_list[idx]



if __name__ == "__main__" :
    data_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/dataset"
    dataset = HydrogelDataset(data_dir, add_targets=True, split_frames=True, add_noise = True, time_window = 2)
    data = dataset[0]
    print(len(dataset))