"""Training and embedding export routines for ACE."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import ContrastiveMAMLoss, SupConLoss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    supcon_loss_fn: nn.Module,
    mam_loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    supcon_weight: float,
    mam_weight: float,
    show_progress_bars: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_supcon_loss = 0.0
    total_mam_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False, disable=not show_progress_bars):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()
        user_embedding, reconstruction_logits = model(
            batch["loc_embeddings"],
            batch["start_hour_indices"],
            batch["start_minute_indices"],
            batch["dur_indices"],
            batch["dow"],
            batch["padding_mask"],
            batch["mam_mask"],
        )
        supcon_loss = supcon_loss_fn(user_embedding, batch["labels"])
        mam_loss = mam_loss_fn(reconstruction_logits, batch["loc_embeddings"], batch["mam_mask"])
        loss = supcon_weight * supcon_loss + mam_weight * mam_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_supcon_loss += supcon_loss.item()
        total_mam_loss += mam_loss.item()

    n_batches = len(dataloader)
    return {
        "loss": total_loss / n_batches if n_batches else 0.0,
        "supcon_loss": total_supcon_loss / n_batches if n_batches else 0.0,
        "mam_loss": total_mam_loss / n_batches if n_batches else 0.0,
    }


def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    supcon_loss_fn: nn.Module,
    mam_loss_fn: nn.Module,
    device: torch.device,
    supcon_weight: float,
    mam_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_supcon_loss = 0.0
    total_mam_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            user_embedding, reconstruction_logits = model(
                batch["loc_embeddings"],
                batch["start_hour_indices"],
                batch["start_minute_indices"],
                batch["dur_indices"],
                batch["dow"],
                batch["padding_mask"],
                batch["mam_mask"],
            )
            supcon_loss = supcon_loss_fn(user_embedding, batch["labels"])
            mam_loss = mam_loss_fn(reconstruction_logits, batch["loc_embeddings"], batch["mam_mask"])
            loss = supcon_weight * supcon_loss + mam_weight * mam_loss
            total_loss += loss.item()
            total_supcon_loss += supcon_loss.item()
            total_mam_loss += mam_loss.item()

    n_batches = len(dataloader)
    return {
        "loss": total_loss / n_batches if n_batches else 0.0,
        "supcon_loss": total_supcon_loss / n_batches if n_batches else 0.0,
        "mam_loss": total_mam_loss / n_batches if n_batches else 0.0,
    }


def run_training_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    paths: dict[str, str],
    config: dict[str, Any],
) -> tuple[nn.Module, float, dict[str, Any]]:
    supcon_loss_fn = SupConLoss()
    mam_loss_fn = ContrastiveMAMLoss()
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_supcon_loss": [],
        "val_supcon_loss": [],
        "train_mam_loss": [],
        "val_mam_loss": [],
        "supcon_weight": config["supcon_weight"],
        "mam_weight": config["mam_weight"],
    }
    show_progress_bars = config.get("show_progress_bars", True)

    for epoch in range(config["num_epochs"]):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            supcon_loss_fn,
            mam_loss_fn,
            optimizer,
            device,
            config["supcon_weight"],
            config["mam_weight"],
            show_progress_bars,
        )
        val_metrics = eval_one_epoch(
            model,
            val_loader,
            supcon_loss_fn,
            mam_loss_fn,
            device,
            config["supcon_weight"],
            config["mam_weight"],
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_supcon_loss"].append(train_metrics["supcon_loss"])
        history["val_supcon_loss"].append(val_metrics["supcon_loss"])
        history["train_mam_loss"].append(train_metrics["mam_loss"])
        history["val_mam_loss"].append(val_metrics["mam_loss"])
        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={train_metrics['loss']:.4f} "
            f"(SupCon={train_metrics['supcon_loss']:.4f}, MAM={train_metrics['mam_loss']:.4f}), "
            f"Val Loss={val_metrics['loss']:.4f} "
            f"(SupCon={val_metrics['supcon_loss']:.4f}, MAM={val_metrics['mam_loss']:.4f})"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(model.state_dict(), paths["save_path"])
            print("  (Best model saved)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_wts)
    return model, best_val_loss, history


def mean_or_none(embeddings: list[torch.Tensor]) -> torch.Tensor | None:
    if len(embeddings) == 0:
        return None
    return torch.cat(embeddings, dim=0).mean(dim=0)


def generate_user_embeddings(
    model: nn.Module,
    activity_chains: list[pd.DataFrame],
    user_to_indices: dict[int, list[int]],
    device: torch.device,
    loc_embedding_type: str,
    minute_interval: int = 15,
    duration_interval: int = 5,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    show_progress_bars = getattr(model, "show_progress_bars", True)
    model.eval()
    user_embeddings: dict[int, dict[str, Any]] = {}
    all_embeddings: list[dict[str, Any]] = []

    with torch.no_grad():
        for uid, chain_indices in tqdm(
            user_to_indices.items(),
            desc="Generating embeddings",
            disable=not show_progress_bars,
        ):
            weekday_chain_embeddings = []
            weekend_chain_embeddings = []
            all_user_chain_embeddings = []

            for chain_idx in chain_indices:
                act_df = activity_chains[chain_idx]
                loc_embeddings = torch.tensor(
                    np.vstack(act_df[f"{loc_embedding_type}_embedding"].values),
                    dtype=torch.float,
                ).unsqueeze(0).to(device)
                start_hour = torch.tensor(act_df["hour"].values + 1, dtype=torch.long).unsqueeze(0).to(device)
                start_minute = torch.tensor(
                    (act_df["minute"].values // minute_interval).astype(np.int64) + 1,
                    dtype=torch.long,
                ).unsqueeze(0).to(device)
                duration_indices = torch.tensor(
                    (act_df["duration_m"].values // duration_interval).astype(np.int64) + 1,
                    dtype=torch.long,
                ).unsqueeze(0).to(device)
                dow = int(act_df["dow"].iloc[0])
                is_weekend = int(dow >= 5)
                dow_tensor = torch.tensor([dow], dtype=torch.long, device=device)

                lengths = torch.tensor([len(act_df)], dtype=torch.long, device=device)
                max_len = loc_embeddings.size(1)
                range_matrix = torch.arange(max_len, device=device).unsqueeze(0)
                padding_mask = range_matrix >= lengths.unsqueeze(1)
                mam_mask = torch.zeros_like(padding_mask, dtype=torch.bool)

                user_emb, _ = model(
                    loc_embeddings,
                    start_hour,
                    start_minute,
                    duration_indices,
                    dow_tensor,
                    padding_mask,
                    mam_mask,
                )
                embedding = user_emb.cpu()
                all_user_chain_embeddings.append(embedding)
                if is_weekend:
                    weekend_chain_embeddings.append(embedding)
                else:
                    weekday_chain_embeddings.append(embedding)
                all_embeddings.append(
                    {
                        "user_id_num": uid,
                        "chain_idx": chain_idx,
                        "dow": dow,
                        "is_weekend": is_weekend,
                        "embedding": embedding.squeeze(0),
                    }
                )

            user_embeddings[uid] = {
                "overall_embedding": mean_or_none(all_user_chain_embeddings),
                "weekday_embedding": mean_or_none(weekday_chain_embeddings),
                "weekend_embedding": mean_or_none(weekend_chain_embeddings),
                "n_total_chains": len(all_user_chain_embeddings),
                "n_weekday_chains": len(weekday_chain_embeddings),
                "n_weekend_chains": len(weekend_chain_embeddings),
            }

    return user_embeddings, all_embeddings
