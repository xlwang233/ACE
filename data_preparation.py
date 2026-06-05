"""Data loading and activity-chain preparation for ACE."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm


def load_location_encoder(ckpt_path: str, device: torch.device) -> nn.Module:
    from model import LocCLIPLightning

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt["hyper_parameters"]["config"]["model"]["reconstruct"] = False
    ckpt["hyper_parameters"]["config"]["data"]["loc_type"] = "x-y"

    lightning_model = LocCLIPLightning(**ckpt["hyper_parameters"]).to(device)
    lightning_model.load_state_dict(ckpt["state_dict"])
    lightning_model.eval()
    loc_encoder = lightning_model.model.loc_enc
    for param in loc_encoder.parameters():
        param.requires_grad = False
    return loc_encoder


def load_and_prepare_stays(config: dict[str, Any], device: torch.device) -> tuple[pd.DataFrame, dict[int, str]]:
    show_progress_bars = config.get("show_progress_bars", True)
    df = pd.read_csv(config["stay_data_path"])
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["start_time"]).dt.date.astype(str)
    df = df.sort_values(by=["user_id", "date", "start_time"])

    loc_embedding_type = config["loc_embedding_type"]
    if loc_embedding_type == "calliper":
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"], crs=4326),
        )
        gdf = gdf.to_crs(config.get("projected_crs", "EPSG:27700"))
        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y
        df = gdf.drop(columns=["geometry"]).sort_values(by=["user_id", "date", "start_time"])

        loc_encoder = load_location_encoder(config["pretrained_loc_encoder_ckpt_path"], device)
        coords = torch.tensor(df[["x", "y"]].values, dtype=torch.float64, device=device)
        with torch.no_grad():
            embeddings = loc_encoder(coords).cpu().numpy()
        df["calliper_embedding"] = embeddings.tolist()
    elif loc_embedding_type == "aether":
        import rasterio

        print("Using AETHER embeddings from the GeoTIFF file...")
        with rasterio.open(config["aether_tif_path"]) as src:
            data = src.read()
        embeddings = []
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Loading AETHER embeddings",
            disable=not show_progress_bars,
        ):
            embeddings.append(data[:, int(row["row"]), int(row["col"])])
        df["aether_embedding"] = embeddings
    else:
        raise ValueError(f"Unsupported loc_embedding_type: {loc_embedding_type}")

    user_id_to_num = {uid: idx for idx, uid in enumerate(df["user_id"].unique())}
    df["user_id_num"] = df["user_id"].map(user_id_to_num)
    user_id_num_to_str = {v: k for k, v in user_id_to_num.items()}

    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["hour"] = df["start_time"].dt.hour
    df["minute"] = df["start_time"].dt.minute
    if "total_time_s" in df.columns:
        df["duration_s"] = df["total_time_s"]
    if "duration_s" not in df.columns:
        df["duration_s"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    df["duration_s"] = df["duration_s"].astype(int)
    df["duration_m"] = df["duration_s"] // 60
    df["dow"] = df["start_time"].dt.dayofweek

    return df, user_id_num_to_str


def prepare_activity_chain_index(df: pd.DataFrame) -> tuple[list[pd.DataFrame], dict[int, list[int]]]:
    print("Grouping data by user and date...")
    activity_chains: list[pd.DataFrame] = []
    user_to_indices: dict[int, list[int]] = defaultdict(list)

    for current_idx, ((uid_num, _), group) in enumerate(df.groupby(["user_id_num", "date"])):
        uid_num = int(uid_num)
        activity_chains.append(group)
        user_to_indices[uid_num].append(current_idx)

    print(f"Total activity chains: {len(activity_chains)}")
    print(f"Total unique users: {len(user_to_indices)}")
    return activity_chains, user_to_indices
