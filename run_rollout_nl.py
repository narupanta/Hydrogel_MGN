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
        gt_world_pos = data[t].target_world_pos.to(device)
        with torch.no_grad():
            pred_world_pos, pred_pvf = model.predict(curr_graph.to(device))  # (num_nodes, time_window)
            # dirichlet_nodes = curr_graph.node_type == 1
            # pred_world_pos[-1, dirichlet_nodes, :] = gt_world_pos[-1, dirichlet_nodes, :]
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
    data_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/dataset/nl_dataset"
    output_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/rollout/{test_on}"
    paraview_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/rollout/{test_on}"
    time_window = 1
    model_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/trained_model/2025-05-31T12h04m44s/model_checkpoint"
    dataset = HydrogelNonLinearDataset(data_dir, add_targets= True, split_frames=True, add_noise = False, time_window=time_window)
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

