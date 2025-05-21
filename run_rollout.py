import numpy as np
from torch_geometric.data import Data
import json
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
from core.datasetclass import HydrogelDataset
from core.model_graphnet import EncodeProcessDecode
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
    disp_mse_tensor = torch.zeros((1,), device = model._device)
    chem_pot_mse_tensor = torch.zeros((1,), device = model._device)
    pred_disp_list = [curr_graph.displacement.to(model._device)]
    gt_disp_list = [curr_graph.displacement.to(model._device)]
    pred_chem_pot_list = [curr_graph.chem_pot.to(model._device)]
    gt_chem_pot_list = [curr_graph.chem_pot.to(model._device)]
    progress = tqdm(range(0, timesteps - time_window, time_window), desc="Rollout")

    for t in progress: 
        # === Predict next time_window temperatures ===
        with torch.no_grad():
            pred_disp, pred_chem_pot = model.predict(curr_graph.to(device))  # (num_nodes, time_window)
            curr_graph.displacement = pred_disp[-1, :, :].clone()
            curr_graph.chem_pot = pred_chem_pot[-1, :, :].clone()  # use last timestep prediction for next input

        # === Ground truth from data[t+1] to data[t+time_window] ===
        gt_disp = data[t].target_displacement.to(device)
        gt_chem_pot = data[t].target_chem_pot.to(device)
        window_disp_error = torch.sum((pred_disp - gt_disp) ** 2, dim = 2)
        window_disp_mse = torch.mean(window_disp_error, dim = 1)

        window_chem_pot_error = torch.sum((pred_chem_pot - gt_chem_pot) ** 2, dim = 2)
        window_chem_pot_mse = torch.mean(window_chem_pot_error, dim = 1)

        disp_mse_tensor = torch.cat([disp_mse_tensor, window_disp_mse])  # (time_window,)
        chem_pot_mse_tensor = torch.cat([chem_pot_mse_tensor, window_chem_pot_mse])
        pred_disp_list.append(pred_disp)
        gt_disp_list.append(gt_disp)

        pred_chem_pot_list.append(pred_chem_pot)
        gt_chem_pot_list.append(gt_chem_pot)

        # print(f"[t={t}] | MSE: {torch.mean(window_disp_mse):.4f}")

    return dict(
        mesh_pos=initial_state.mesh_pos,
        node_type=initial_state.node_type,
        cells=initial_state.cells,
        predict_displacement=pred_disp_list,
        gt_displacement=gt_disp_list,
        predict_chem_pot=pred_chem_pot_list,
        gt_chem_pot=gt_chem_pot_list,
        disp_mse = disp_mse_tensor,
        chem_pot_mse = chem_pot_mse_tensor
    )

if __name__ == "__main__" :
    test_on = "poc"
    data_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/dataset"
    output_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/rollout/{test_on}"
    paraview_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/rollout/{test_on}"
    time_window = 1
    model_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/trained_model/2025-05-15T14h22m29s/model_checkpoint"
    dataset = HydrogelDataset(data_dir, add_targets= True, split_frames=True, add_noise = False, time_window=time_window)
    data = dataset[0]
    model = EncodeProcessDecode(node_feature_size = 5,
                                mesh_edge_feature_size = 7,
                                output_size = 3,
                                latent_size = 128,
                                timestep=1e-5,
                                time_window=time_window,
                                device=device,
                                message_passing_steps = 15)
    model.to(device)
    model.load_model(model_dir)
    model.eval()
    model = torch.compile(model)
    output = rollout(model, data, time_window)