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
from run_rollout_nl import rollout
# def load_dataset( , add_target = False, add_noise = False, split = False) :
#     file_path = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/npz_files/weld_fem_60mm.npz"
#     dataset = np.load(file_path)

#     data = Data()
#     return 

# def learner() :
    
device = "cuda"

if __name__ == "__main__" :
    
    data_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/dataset/nl_dataset"
    output_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/trained_model"
    run_dir = prepare_directories(output_dir)
    model_dir = os.path.join(run_dir, 'model_checkpoint')
    logs_dir = os.path.join(run_dir, "logs")
    logger_setup(os.path.join(logs_dir, "logs.txt"))
    logger = logging.getLogger()
    time_window = 1
    dataset = HydrogelNonLinearDataset(data_dir, add_targets= True, split_frames=True, add_noise = True, time_window = time_window)
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
    num_epochs = 2
    traj_pass = 10
    test_round = 15
    train_loss_per_epochs = []
    is_accumulate_normalizer_phase = True
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0
        val_total_loss, val_disp_loss, val_pvf_loss = 0, [], []
        for traj_idx, trajectory in enumerate(dataset):  # assuming dataset.trajectories exists
            traj_total_loss, traj_disp_loss, traj_pvf_loss = 0, 0, 0
            train_loader = DataLoader(trajectory, batch_size=1, shuffle=True)
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

            for idx_traj, batch in loop:
                batch = batch.to(device)
                optimizer.zero_grad()
                predictions = model(batch)
                total_loss, disp_loss, pvf_loss= model.loss(predictions, batch)

                if not is_accumulate_normalizer_phase:
                    total_loss.backward()
                    optimizer.step()
                    traj_total_loss += total_loss.item()
                    traj_disp_loss += disp_loss.item()
                    traj_pvf_loss += pvf_loss.item()
                    loop.set_description(f"Epoch {epoch + 1} Traj {traj_idx + 1}/{len(dataset)}")
                    loop.set_postfix({"Total Loss": f"{total_loss.item():.4f}",
                                      "Total Disp Loss": f"{disp_loss.item():.4f}",
                                      "Total PVF Loss": f"{pvf_loss.item():.4f}"})
            if (traj_pass > 0) & is_accumulate_normalizer_phase:
                traj_pass -= 1
            else :
                is_accumulate_normalizer_phase = False
            train_total_loss += traj_total_loss
            logger.info(f"Epoch {epoch+1}, Trajectory {traj_idx+1}: Train Total Loss: {traj_total_loss:.4f}, Train Disp Loss: {traj_disp_loss:.4f}, Train PVF Loss: {traj_pvf_loss:.4f}")
            if test_round == 0 :
                break
            else :
                test_round -= 1

        if not is_accumulate_normalizer_phase:
            if epoch%20 == 0 :
                for traj_idx, trajectory in enumerate(dataset):
                    output = rollout(model, trajectory, time_window)
                    val_disp_loss = torch.mean(output['disp_mse'])
                    val_pvf_loss = torch.mean(output['pvf_mse'])
                    val_total_loss += val_disp_loss + val_pvf_loss
                    logger.info(f"Epoch {epoch + 1}, Trajectory {traj_idx + 1}: Rollout MSE = {val_disp_loss + val_pvf_loss:.6f}, Rollout Disp MSE = {val_disp_loss:.6f}, Rollout Chem MSE = {val_pvf_loss:.6f}")

                avg_train_loss = train_total_loss / len(dataset)
                avg_rollout_loss = val_total_loss / len(dataset)
        
            model.save_model(model_dir)
            torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_state_dict.pth"))
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, train loss: {avg_train_loss:.4f}, rollout loss: {avg_rollout_loss:.4e}")
            print(f"Epoch {epoch + 1}/{num_epochs}, train loss: {avg_train_loss:.4f}, rollout loss: {avg_rollout_loss:.4e}")

