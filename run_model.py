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
from run_rollout import rollout
# def load_dataset( , add_target = False, add_noise = False, split = False) :
#     file_path = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/npz_files/weld_fem_60mm.npz"
#     dataset = np.load(file_path)

#     data = Data()
#     return 

# def learner() :
    
device = "cuda"

if __name__ == "__main__" :
    
    data_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/dataset/linear_hydrogel"
    output_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/trained_model"
    run_dir = prepare_directories(output_dir)
    model_dir = os.path.join(run_dir, 'model_checkpoint')
    logs_dir = os.path.join(run_dir, "logs")
    logger_setup(os.path.join(logs_dir, "logs.txt"))
    logger = logging.getLogger()
    time_window = 1
    dataset = HydrogelDataset(data_dir, add_targets= True, split_frames=True, add_noise = True, time_window = time_window)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
    num_epochs = 100
    train_loss_per_epochs = []
    is_accumulate_normalizer_phase = True
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_total_loss, train_disp_loss, train_chem_loss = 0, 0, 0
        val_total_loss, val_disp_loss, val_chem_loss = 0, 0, 0
        for traj_idx, trajectory in enumerate(dataset):  # assuming dataset.trajectories exists
            train_loader = DataLoader(trajectory, batch_size=1, shuffle=True)
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

            for idx_traj, batch in loop:
                batch = batch.to(device)
                optimizer.zero_grad()
                predictions = model(batch)
                total_loss, disp_loss, chem_loss= model.loss(predictions, batch)

                if not is_accumulate_normalizer_phase:
                    total_loss.backward()
                    optimizer.step()
                    train_total_loss += total_loss.item()
                    train_disp_loss += disp_loss.item()
                    train_chem_loss += chem_loss.item()
                    loop.set_description(f"Epoch {epoch + 1} Traj {traj_idx + 1}/{len(dataset)}")
                    loop.set_postfix({"Total Loss": f"{total_loss.item():.4f}",
                                      "Total Disp Loss": f"{disp_loss.item():.4f}",
                                      "Total Chem Loss": f"{chem_loss.item():.4f}"})
            if not is_accumulate_normalizer_phase:
                output = rollout(model, trajectory, time_window)
                val_disp_loss = torch.mean(output['disp_mse'])
                val_chem_loss = torch.mean(output['chem_pot_mse'])
                val_total_loss += val_disp_loss + val_chem_loss
                logger.info(f"Epoch {epoch + 1}, Trajectory {traj_idx + 1}: Rollout MSE = {val_total_loss:.6f}, Rollout Disp MSE = {val_total_loss:.6f}, Rollout Chem MSE = {val_total_loss:.6f}")
            else:
                is_accumulate_normalizer_phase = False
        logger.info(f"Epoch {epoch+1}, Trajectory {traj_idx+1}: Train Total Loss: {train_total_loss:.4f}, Train Disp Loss: {train_disp_loss:.4f}, Train Chem Loss: {train_chem_loss:.4f}")
        if not is_accumulate_normalizer_phase:
            avg_train_loss = train_total_loss / len(dataset)
            avg_rollout_loss = val_total_loss / len(dataset)
            train_loss_per_epochs.append(avg_train_loss)
    
            if avg_rollout_loss < best_val_loss:
                best_val_loss = avg_rollout_loss
                model.save_model(model_dir)
                torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_state_dict.pth"))
            print(f"Epoch {epoch + 1}/{num_epochs}, train loss: {avg_train_loss:.4f}, rollout loss: {avg_rollout_loss:.4e}")

