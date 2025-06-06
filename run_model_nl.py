import os
import logging
import yaml
from pathlib import Path
from tqdm import tqdm

import torch
from torch_geometric.loader import DataLoader

from core.datasetclass import HydrogelNonLinearDataset
from core.model_graphnet_nl import EncodeProcessDecode
from core.utils import prepare_directories, logger_setup
from run_rollout_nl import rollout


def load_config(path="train_config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg, device):
    m_cfg = cfg["model"]
    model = EncodeProcessDecode(
        node_feature_size=m_cfg["node_feature_size"],
        mesh_edge_feature_size=m_cfg["mesh_edge_feature_size"],
        output_size=m_cfg["output_size"],
        latent_size=m_cfg["latent_size"],
        timestep=m_cfg["timestep"],
        time_window=cfg["training"]["time_window"],
        device=device,
        message_passing_steps=cfg["training"]["message_passing_steps"],
    )
    return model.to(device)


def train(model, train_dataset, val_dataset, optimizer, run_dir, model_dir, logs_dir, cfg, device):
    logger_setup(os.path.join(logs_dir, "logs.txt"))
    logger = logging.getLogger()

    best_val_loss = float("inf")
    if cfg["paths"].get("model_dir") :
        traj_pass = 0
        is_accumulate_phase = False
    else :
        traj_pass = cfg["training"]["trial_passes"]
        is_accumulate_phase = True
    num_epochs = cfg["training"]["num_epochs"]
    time_window = cfg["training"]["time_window"]

    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0
        logger.info(f"==== Epoch {epoch + 1} ====")

        for traj_idx, trajectory in enumerate(train_dataset):
            traj_total_loss, traj_disp_loss, traj_pvf_loss = 0, 0, 0
            train_loader = DataLoader(trajectory, batch_size=1, shuffle=True)
            loop = tqdm(train_loader, leave=False)

            for batch in loop:
                batch = batch.to(device)
                optimizer.zero_grad()
                predictions = model(batch)
                total_loss, disp_loss, pvf_loss = model.loss(predictions, batch)

                if not is_accumulate_phase:
                    total_loss.backward()
                    optimizer.step()
                    traj_total_loss += total_loss.item()
                    traj_disp_loss += disp_loss.item() 
                    traj_pvf_loss += pvf_loss.item()
                    loop.set_description(f"Epoch {epoch + 1}, Traj {traj_idx + 1}")
                    loop.set_postfix({
                        "Loss": f"{total_loss.item():.4f}",
                        "Disp": f"{disp_loss.item():.4f}",
                        "PVF": f"{pvf_loss.item():.4f}"
                    })

            if traj_pass > 0 and is_accumulate_phase:
                traj_pass -= 1
            else:
                is_accumulate_phase = False

            train_total_loss += traj_total_loss
            logger.info(
                f"Trajectory {traj_idx + 1}: Train Loss: {traj_total_loss:.4f}, Disp Loss: {traj_disp_loss:.4f}, PVF Loss: {traj_pvf_loss:.4f}"
            )

        # === Validation ===
        if not is_accumulate_phase:
            val_total_loss = 0.0
            for traj_idx, trajectory in enumerate(val_dataset):
                output = rollout(model, trajectory, time_window)
                disp_mse = torch.mean(output["disp_mse"])
                pvf_mse = torch.mean(output["pvf_mse"])
                val_loss = disp_mse + pvf_mse
                val_total_loss += val_loss.item()

                logger.info(
                    f"Val Traj {traj_idx + 1}: Rollout MSE: {val_loss:.6e}, Disp MSE: {disp_mse:.6e}, PVF MSE: {pvf_mse:.6e}"
                )

            avg_train_loss = train_total_loss / len(train_dataset)
            avg_val_loss = val_total_loss / len(val_dataset)

            logger.info(f"Epoch {epoch + 1} Summary - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.6e}")
            print(f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.6e}")

            if avg_val_loss < best_val_loss:
                model.save_model(model_dir)
                torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_state_dict.pth"))
                best_val_loss = avg_val_loss
                logger.info("Checkpoint saved (best model so far).")


def load_checkpoint_if_available(model, optimizer, model_dir):
    optim_path = os.path.join(model_dir, "optimizer_state_dict.pth")

    if os.path.exists(model_dir) and os.path.exists(optim_path):
        model.load_model(model_dir)
        optimizer.load_state_dict(torch.load(optim_path))
        print(f"Resumed training from checkpoint at: {model_dir}")
        return True
    else:
        print(f"Model path not found in: {model_dir}. Starting fresh.")
        return False

def setup_training_environment(cfg):
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    paths = cfg["paths"]

    # Decide run_dir and logging/model paths
    if paths.get("model_dir"):  # Resume from checkpoint
        run_dir = prepare_directories(paths["output_dir"])
        model_dir = paths["model_dir"]
        logs_dir = os.path.join(run_dir, "logs")

        config_summary_path = os.path.join(run_dir, "config_summary.yaml")
        with open(config_summary_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    else:  # Start fresh training
        run_dir = prepare_directories(paths["output_dir"])
        model_dir = os.path.join(run_dir, "model_checkpoint")
        logs_dir = os.path.join(run_dir, "logs")

        # Save a clean summary of the config at start of training
        config_summary_path = os.path.join(run_dir, "config_summary.yaml")
        with open(config_summary_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    train_dataset = HydrogelNonLinearDataset(
        paths["data_dir"],
        add_targets=True,
        split_frames=True,
        add_noise=True,
        time_window=cfg["training"]["time_window"],
        target_config=cfg["training"]["target_config"]
    )

    val_dataset = HydrogelNonLinearDataset(
        paths["data_dir"],
        add_targets=True,
        split_frames=True,
        add_noise=False,
        time_window=cfg["training"]["time_window"],
        target_config = {"world_pos": {"noise": 0.000}, "pvf": {"noise": 0.000}}
    )
    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )

    resumed = False
    if paths.get("model_dir"):
        resumed = load_checkpoint_if_available(model, optimizer, model_dir)

    return model, optimizer, train_dataset, val_dataset, run_dir, model_dir, logs_dir, cfg, device


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train EncodeProcessDecode model")
    parser.add_argument('--config', type=str, default="train_config.yml", help="Path to the config YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, optimizer, train_dataset, val_dataset, run_dir, model_dir, logs_dir, cfg, device = setup_training_environment(cfg)

    train(model, train_dataset, val_dataset, optimizer, run_dir, model_dir, logs_dir, cfg, device)

if __name__ == "__main__":
    main()

