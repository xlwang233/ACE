"""Experiment helpers for ACE training runs."""

from __future__ import annotations

import atexit
import datetime as dt
import json
import os
import random
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def configure_experiment_logging(log_path: str) -> None:
    """Mirror stdout/stderr to the experiment-local run.log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    def restore_streams() -> None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(restore_streams)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {device_name}, but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def make_ace_experiment_paths(
    save_dir: str,
    city: str,
    loc_embedding_type: str,
    experiment_name: str,
    create_dirs: bool = True,
) -> dict[str, str]:
    """Build and optionally create the standard ACE experiment artifact paths."""
    paths = {
        "experiment_name": experiment_name,
        "experiment_dir": os.path.join(save_dir, "experiments", experiment_name),
    }
    paths.update(
        {
            "model_dir": os.path.join(paths["experiment_dir"], "models"),
            "log_dir": os.path.join(paths["experiment_dir"], "logs"),
            "embedding_dir": os.path.join(paths["experiment_dir"], "embeddings"),
            "clustering_dir": os.path.join(paths["experiment_dir"], "clustering"),
            "figure_dir": os.path.join(paths["experiment_dir"], "figures"),
        }
    )
    paths.update(
        {
            "save_path": os.path.join(paths["model_dir"], f"best_ace_model_{city}_{loc_embedding_type}.pt"),
            "training_loss_plot_path": os.path.join(paths["figure_dir"], "training_losses.png"),
            "training_history_path": os.path.join(paths["log_dir"], "training_history.csv"),
            "experiment_config_path": os.path.join(paths["log_dir"], "experiment_config.json"),
            "run_log_path": os.path.join(paths["log_dir"], "run.log"),
            "embedding_save_path": os.path.join(
                paths["embedding_dir"], f"user_embeddings_{city}_{loc_embedding_type}.pt"
            ),
            "chain_embedding_save_path": os.path.join(
                paths["embedding_dir"], f"chain_embeddings_{city}_{loc_embedding_type}.pt"
            ),
            "metadata_save_path": os.path.join(paths["embedding_dir"], "user_embeddings_metadata.pt"),
            "clustering_results_path": os.path.join(
                paths["clustering_dir"], f"user_clustering_results_{city}_{loc_embedding_type}.csv"
            ),
            "user_id_mapping_path": os.path.join(paths["experiment_dir"], "user_id_mapping.csv"),
        }
    )
    if create_dirs:
        for dir_key in ["experiment_dir", "model_dir", "log_dir", "embedding_dir", "clustering_dir", "figure_dir"]:
            os.makedirs(paths[dir_key], exist_ok=True)
    return paths


def make_experiment_name(config: dict[str, Any]) -> str:
    explicit_name = config.get("experiment_name")
    if explicit_name:
        return explicit_name

    model_config = config["model"]
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{timestamp}_"
        f"mam-{model_config.get('mam_mask_mode', 'location_only')}_"
        f"emb-{model_config.get('user_embedding_mode', 'prompt')}_"
        f"prompt-{int(model_config.get('use_prompt_token', True))}_"
        f"mamp{config.get('mam_mask_prob', 0.15):g}_"
        f"minmask{config.get('min_mam_masks_per_chain', 0)}_"
        f"sw{config.get('supcon_weight', 1.0):g}_mamw{config.get('mam_weight', 1.0):g}"
    )


def plot_training_history(history: dict[str, Any], path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    if not history["train_loss"]:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Weighted Loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Validation Weighted Loss")
    plt.plot(epochs, history["train_supcon_loss"], alpha=0.5, label="Train SupCon Loss")
    plt.plot(epochs, history["val_supcon_loss"], alpha=0.5, label="Validation SupCon Loss")
    plt.plot(epochs, history["train_mam_loss"], alpha=0.5, label="Train MAM Loss")
    plt.plot(epochs, history["val_mam_loss"], alpha=0.5, label="Validation MAM Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
