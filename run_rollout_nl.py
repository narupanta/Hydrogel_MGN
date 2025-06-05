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


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

device = "cuda"
def rollout(model, data, time_window, device="cuda"):
    """Roll out predictions by stepping `time_window` ahead at each iteration."""
    data = list(data)
    initial_state = data[0]
    timesteps = len(data)

    curr_graph = initial_state.clone()
    pred_world_pos_list = [curr_graph.world_pos.to(device).unsqueeze(0)]
    gt_world_pos_list = [curr_graph.world_pos.to(device).unsqueeze(0)]
    pred_pvf_list = [curr_graph.pvf.to(device).unsqueeze(0)]
    gt_pvf_list = [curr_graph.pvf.to(device).unsqueeze(0)]

    disp_mse_list = []
    pvf_mse_list = []

    progress = tqdm(range(0, timesteps, time_window), desc="Rollout")

    for t in progress:
        # === Model predicts time_window steps ===
        with torch.no_grad():
            pred_world_pos, pred_pvf = model.predict(curr_graph.to(device))
            curr_graph.world_pos = pred_world_pos[-1]  # advance to last predicted
            curr_graph.pvf = pred_pvf[-1]

        # === Get ground truth ===
        if t < len(data):  # valid ground truth frame exists
            gt_world_pos = data[t].target_world_pos.to(device)
            gt_pvf = data[t].target_pvf.to(device)

            # MSE over prediction window
            world_pos_mse = ((pred_world_pos - gt_world_pos) ** 2).sum(dim=-1).mean(dim=-1)
            pvf_mse = ((pred_pvf - gt_pvf) ** 2).sum(dim=-1).mean(dim=-1)

            disp_mse_list.append(world_pos_mse)
            pvf_mse_list.append(pvf_mse)

            gt_world_pos_list.append(gt_world_pos)
            gt_pvf_list.append(gt_pvf)

        pred_world_pos_list.append(pred_world_pos)
        pred_pvf_list.append(pred_pvf)

        progress.set_description(f"[t={t}] | disp_mse: {world_pos_mse.mean():.4e}")

    return {
        "mesh_pos": initial_state.mesh_pos,
        "node_type": initial_state.node_type,
        "cells": initial_state.cells,
        "predict_displacement": torch.cat(pred_world_pos_list, dim=0),
        "gt_displacement": torch.cat(gt_world_pos_list, dim=0),
        "predict_pvf": torch.cat(pred_pvf_list, dim=0),
        "gt_pvf": torch.cat(gt_pvf_list, dim=0),
        "disp_mse": torch.cat(disp_mse_list),
        "pvf_mse": torch.cat(pvf_mse_list),
    }


if __name__ == "__main__" :
    test_on = "nonlinear_poc"
    # data_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/dataset/nl_dataset"
    data_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/testcases"
    output_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/rollout/{test_on}"
    paraview_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/rollout/{test_on}"
    time_window = 2
    model_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/trained_model/2025-06-05T15h45m54s/model_checkpoint"
    dataset = HydrogelNonLinearDataset(data_dir, add_targets= True, split_frames=True, add_noise = False, time_window=time_window, target_config = {"world_pos": {"noise": 0.000}, "pvf": {"noise": 0.000}})
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

    # Setup
    timestep = 150
    mesh_pos = output["mesh_pos"].detach().cpu().numpy()
    cells = output["cells"].detach().cpu().numpy()
    pred_displacement = output["predict_displacement"][timestep].squeeze(0).detach().cpu().numpy()
    gt_displacement = output["gt_displacement"][timestep].squeeze(0).detach().cpu().numpy()
    predict_pvf = output["predict_pvf"][timestep].squeeze(0).detach().cpu().numpy()
    gt_pvf = output["gt_pvf"][timestep].squeeze(0).detach().cpu().numpy()

    # Deformed positions
    deformed_nodes_gt = gt_displacement
    deformed_nodes_pred = pred_displacement

    # Triangulations
    triang_gt = tri.Triangulation(deformed_nodes_gt[:, 0], deformed_nodes_gt[:, 1], cells)
    triang_pred = tri.Triangulation(deformed_nodes_pred[:, 0], deformed_nodes_pred[:, 1], cells)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    # Ground truth
    tpc1 = axs[0].tripcolor(triang_gt, gt_pvf.squeeze(), shading='gouraud', cmap='viridis', edgecolors='k', linewidth=0.2)
    axs[0].triplot(mesh_pos[:, 0], mesh_pos[:, 1], cells, color='grey', linewidth=0.5, label='Original Mesh')
    axs[0].set_title(f'Ground Truth Polymer Volume Fraction\n(Timestep {timestep})')
    axs[0].axis('equal')
    axs[0].grid(True)
    axs[0].legend()
    fig.colorbar(tpc1, ax=axs[0], label='GT Polymer Volume Fraction')

    # Prediction
    tpc2 = axs[1].tripcolor(triang_pred, predict_pvf.squeeze(), shading='gouraud', cmap='viridis', edgecolors='k', linewidth=0.2)
    axs[1].triplot(mesh_pos[:, 0], mesh_pos[:, 1], cells, color='grey', linewidth=0.5, label='Original Mesh')
    axs[1].set_title(f'Predicted Polymer Volume Fraction\n(Timestep {timestep})')
    axs[1].axis('equal')
    axs[1].grid(True)
    axs[1].legend()
    fig.colorbar(tpc2, ax=axs[1], label='Pred Polymer Volume Fraction')

    plt.suptitle("Deformed Mesh Comparison: Predicted vs Ground Truth", fontsize=16)
    plt.show()
    import meshio

    # for timestep in range(len(output["predict_displacement"])):
    #     displacement = output["predict_displacement"][timestep].squeeze(0).detach().cpu().numpy()
    #     pvf = output["predict_pvf"][timestep].squeeze(0).detach().cpu().numpy()

    #     meshio.write_points_cells(
    #         f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/rollout/nonlinear_poc/pred_{timestep:04d}.vtu",
    #         points= displacement,
    #         cells=[("triangle", cells)],
    #         point_data={"pvf": pvf},
    #     )
    import os
    import numpy as np
    import meshio

    # Output folders
    base_dir = "/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/rollout/nonlinear_rollout"
    pred_dir = os.path.join(base_dir, "pred")
    gt_dir = os.path.join(base_dir, "gt")

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Ensure displacements are padded to 3D
    def pad_to_3d(arr):
        if arr.shape[1] == 2:
            return np.hstack([arr, np.zeros((arr.shape[0], 1))])
        return arr

    for timestep in range(len(output["predict_displacement"])):
        # Base mesh
        base_pos = mesh_pos

        # Predicted
        pred_disp = output["predict_displacement"][timestep].squeeze(0).detach().cpu().numpy()
        pred_pvf = output["predict_pvf"][timestep].squeeze(0).detach().cpu().numpy()

        # Ground truth
        gt_disp = output["gt_displacement"][timestep].squeeze(0).detach().cpu().numpy()
        gt_pvf = output["gt_pvf"][timestep].squeeze(0).detach().cpu().numpy()

        # Pad displacements to 3D if needed
        pred_disp_3d = pad_to_3d(pred_disp)
        gt_disp_3d = pad_to_3d(gt_disp)

        # Write predicted file
        meshio.write_points_cells(
            os.path.join(pred_dir, f"pred_{timestep:04d}.vtu"),
            points=pred_disp_3d,
            cells=[("triangle", cells)],
            point_data={"pvf": pred_pvf}
        )

        # Write ground truth file
        meshio.write_points_cells(
            os.path.join(gt_dir, f"gt_{timestep:04d}.vtu"),
            points=gt_disp_3d,
            cells=[("triangle", cells)],
            point_data={"pvf": gt_pvf}
        )

