import numpy as np
from torch_geometric.data import Data
import json
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
from core.datasetclass import HydrogelNonLinearDataset
from core.model_graphnet_nl import EncodeProcessDecode
import numpy as np
from tqdm import tqdm
from core.utils import * 
import h5py
import meshio
import time
    
device = "cuda"
def run_step(model, graph) :
    with torch.no_grad():
        curr_temp = model.predict(graph.to(device))
        graph.temperature = curr_temp
        return graph

def rollout(model, data, time_window):
    data = [frame for frame in data]
    initial_state = data[0]
    timesteps = len(data)

    rmse_list = []

    curr_graph = initial_state.clone()  # Start with first ground truth graph
    world_pos_mse_tensor = torch.zeros((1,), device = model._device)
    pvf_mse_tensor = torch.zeros((1,), device = model._device)
    pred_world_pos_list = [curr_graph.world_pos.to(model._device)]
    gt_world_pos_list = [curr_graph.world_pos.to(model._device)]
    pred_pvf_list = [curr_graph.pvf.to(model._device)]
    gt_pvf_list = [curr_graph.pvf.to(model._device)]
    progress = tqdm(range(0, timesteps - time_window, time_window), desc="Rollout")

    for t in progress: 
        # === Predict next time_window temperatures ===
        with torch.no_grad():
            pred_world_pos, pred_pvf = model.predict(curr_graph.to(device))  # (num_nodes, time_window)
            curr_graph.world_pos = pred_world_pos[-1, :, :].clone()
            curr_graph.pvf = pred_pvf[-1, :, :].clone()  # use last timestep prediction for next input

        # === Ground truth from data[t+1] to data[t+time_window] ===
        gt_world_pos = data[t].target_world_pos.to(device)
        gt_pvf = data[t].target_pvf.to(device)
        window_world_pos_error = torch.sum((pred_world_pos - gt_world_pos) ** 2, dim = 2)
        window_world_pos_mse = torch.mean(window_world_pos_error, dim = 1)

        window_pvf_error = torch.sum((pred_pvf - gt_pvf) ** 2, dim = 2)
        window_pvf_mse = torch.mean(window_pvf_error, dim = 1)

        world_pos_mse_tensor = torch.cat([world_pos_mse_tensor, window_world_pos_mse])  # (time_window,)
        pvf_mse_tensor = torch.cat([pvf_mse_tensor, window_pvf_mse])
        
        pred_world_pos_list.append(pred_world_pos)
        gt_world_pos_list.append(gt_world_pos)

        pred_pvf_list.append(pred_pvf)
        gt_pvf_list.append(gt_pvf)

        # print(f"[t={t}] | MSE: {torch.mean(window_world_pos_mse):.4f}")

    return dict(
        mesh_pos=initial_state.mesh_pos,
        node_type=initial_state.node_type,
        cells=initial_state.cells,
        predict_displacement=pred_world_pos_list,
        gt_displacement=gt_world_pos_list,
        predict_pvf=pred_pvf_list,
        gt_pvf=gt_pvf_list,
        disp_mse = world_pos_mse_tensor,
        pvf_mse = pvf_mse_tensor
    )

if __name__ == "__main__" :
    test_on = "nonlinear_poc"
    data_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/testcases"
    output_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/rollout/{test_on}"
    paraview_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/rollout/{test_on}"
    time_window = 1
    model_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/trained_model/2025-05-15T14h22m29s/model_checkpoint"
    dataset = HydrogelNonLinearDataset(data_dir, add_targets= True, split_frames=True, add_noise = False, time_window=time_window)
    data = dataset[0]
    model = EncodeProcessDecode(node_feature_size = 4,
                                mesh_edge_feature_size = 7,
                                output_size = 3,
                                latent_size = 128,
                                timestep=1e-5,
                                time_window=time_window,
                                device=device,
                                message_passing_steps = 15)
    model.to(device)
    # model.load_model(model_dir)
    model.eval()
    model = torch.compile(model)
    output = rollout(model, data, time_window)