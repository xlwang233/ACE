#!/usr/bin/env python3
"""Train ACE and export user activity-profile embeddings.

This script reads parameters from ace_config.json by default, trains
ActivityChainEncoder, saves experiment artifacts, reloads the best checkpoint,
and exports overall/weekday/weekend user embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ace_experiment import (
    configure_experiment_logging,
    load_json,
    make_ace_experiment_paths,
    make_experiment_name,
    plot_training_history,
    resolve_device,
    set_seed,
)
from activity_chain_dataset import (
    ActivityChainDataset,
    HybridPKSampler,
    ace_collate_fn,
    build_split_from_users,
    build_weekpart_identity_map,
)
from data_preparation import load_and_prepare_stays, prepare_activity_chain_index
from models import ActivityChainEncoder
from training import generate_user_embeddings, run_training_with_early_stopping


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACE and export user embeddings.")
    parser.add_argument("--config", default="ace_config.json", help="Path to JSON config file.")
    args = parser.parse_args()

    config = load_json(args.config)

    os.makedirs(config["save_dir"], exist_ok=True)
    experiment_name = make_experiment_name(config)
    paths = make_ace_experiment_paths(
        config["save_dir"],
        config["city"],
        config["loc_embedding_type"],
        experiment_name,
    )
    configure_experiment_logging(paths["run_log_path"])
    print(f"Experiment directory: {paths['experiment_dir']}")
    print(f"Run log: {paths['run_log_path']}")

    set_seed(config.get("seed", 101))
    device = resolve_device(config.get("device", "cuda:0"))

    df_stay_filtered, user_id_num_to_str = load_and_prepare_stays(config, device)
    pd.DataFrame(list(user_id_num_to_str.items()), columns=["user_id_num", "user_id"]).to_csv(
        paths["user_id_mapping_path"],
        index=False,
    )

    activity_chains, user_map = prepare_activity_chain_index(df_stay_filtered)

    model = ActivityChainEncoder(config["model"]).to(device)
    model.show_progress_bars = config.get("show_progress_bars", True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    all_users = list(user_map.keys())
    if len(all_users) < 2:
        raise ValueError("Need at least 2 users to create both train and validation sets.")

    rng = random.Random(config.get("split_seed", config.get("seed", 101)))
    rng.shuffle(all_users)
    n_val_users = int(len(all_users) * config["val_ratio"])
    n_val_users = max(1, min(n_val_users, len(all_users) - 1))
    val_users = set(all_users[:n_val_users])
    train_users = set(all_users[n_val_users:])

    train_chains, _ = build_split_from_users(activity_chains, user_map, train_users)
    val_chains, _ = build_split_from_users(activity_chains, user_map, val_users)
    train_dataset = ActivityChainDataset(
        train_chains,
        config["loc_embedding_type"],
        config["minute_interval"],
        config["duration_interval"],
    )
    val_dataset = ActivityChainDataset(
        val_chains,
        config["loc_embedding_type"],
        config["minute_interval"],
        config["duration_interval"],
    )

    train_sampler = HybridPKSampler(
        build_weekpart_identity_map(train_chains),
        batch_size=config["batch_size"],
        single_user_ratio=config["single_user_ratio"],
    )
    val_sampler = HybridPKSampler(
        build_weekpart_identity_map(val_chains),
        batch_size=config["batch_size"],
        single_user_ratio=config["single_user_ratio"],
    )

    def collate_with_masking(batch):
        return ace_collate_fn(
            batch,
            mam_mask_prob=config["mam_mask_prob"],
            min_mam_masks_per_sequence=config["min_mam_masks_per_chain"],
        )

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_with_masking)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_with_masking)

    print(f"Train users: {len(train_users)}, Val users: {len(val_users)}")
    print(f"Train chains: {len(train_dataset)}, Val chains: {len(val_dataset)}")
    print(f"Train batches/epoch: {len(train_loader)}, Val batches/epoch: {len(val_loader)}")

    model, best_val_loss, history = run_training_with_early_stopping(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        paths,
        config,
    )
    if config.get("plot_losses", True):
        plot_training_history(history, paths["training_loss_plot_path"])

    experiment_config = {
        **config,
        "experiment_name": experiment_name,
        "experiment_dir": paths["experiment_dir"],
        "best_val_loss": best_val_loss,
        "best_model_path": paths["save_path"],
        "training_loss_plot_path": paths["training_loss_plot_path"],
        "run_log_path": paths["run_log_path"],
        "train_users": len(train_users),
        "val_users": len(val_users),
        "train_chains": len(train_dataset),
        "val_chains": len(val_dataset),
    }
    pd.DataFrame(history).to_csv(paths["training_history_path"], index_label="epoch")
    with open(paths["experiment_config_path"], "w") as f:
        json.dump(experiment_config, f, indent=2)

    model = ActivityChainEncoder(config["model"]).to(device)
    model.show_progress_bars = config.get("show_progress_bars", True)
    model.load_state_dict(torch.load(paths["save_path"], map_location=device))
    model.eval()
    user_embeddings, all_embeddings = generate_user_embeddings(
        model,
        activity_chains,
        user_map,
        device,
        config["loc_embedding_type"],
        minute_interval=config["minute_interval"],
        duration_interval=config["duration_interval"],
    )

    torch.save(user_embeddings, paths["embedding_save_path"])
    torch.save(all_embeddings, paths["chain_embedding_save_path"])

    sample_user_record = next(iter(user_embeddings.values()))
    metadata = {
        "num_users": len(user_embeddings),
        "embedding_dim": sample_user_record["overall_embedding"].shape[0],
        "embedding_fields": ["overall_embedding", "weekday_embedding", "weekend_embedding"],
        "num_users_with_weekday_embedding": sum(v["weekday_embedding"] is not None for v in user_embeddings.values()),
        "num_users_with_weekend_embedding": sum(v["weekend_embedding"] is not None for v in user_embeddings.values()),
        "user_ids": list(user_embeddings.keys()),
        "experiment_name": experiment_name,
        "experiment_dir": paths["experiment_dir"],
        "model_config": config["model"],
        "best_model_path": paths["save_path"],
    }
    torch.save(metadata, paths["metadata_save_path"])

    print(f"Best ACE checkpoint saved to {paths['save_path']}")
    print(f"Training history saved to {paths['training_history_path']}")
    print(f"User embeddings saved to {paths['embedding_save_path']}")
    print(f"Embedding metadata saved to {paths['metadata_save_path']}")


if __name__ == "__main__":
    main()
