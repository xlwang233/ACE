"""Dataset and batching helpers for ACE activity-chain training."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler


class ActivityChainDataset(Dataset):
    def __init__(
        self,
        activity_chains: list[pd.DataFrame],
        loc_embedding_type: str,
        minute_interval: int = 15,
        duration_interval: int = 5,
    ):
        self.activity_chains = activity_chains
        self.loc_embedding_type = loc_embedding_type
        self.minute_interval = minute_interval
        self.duration_interval = duration_interval

    def __len__(self) -> int:
        return len(self.activity_chains)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        act_df = self.activity_chains[idx]
        loc_embeddings = torch.tensor(
            np.vstack(act_df[f"{self.loc_embedding_type}_embedding"].values),
            dtype=torch.float,
        )
        return {
            "loc_embeddings": loc_embeddings,
            "dow": int(act_df["dow"].iloc[0]),
            "start_hour_indices": torch.tensor(act_df["hour"].values, dtype=torch.long),
            "start_minute_indices": torch.tensor(act_df["minute"].values // self.minute_interval, dtype=torch.long),
            "dur_indices": torch.tensor(act_df["duration_m"].values // self.duration_interval, dtype=torch.long),
            "user_id_num": int(act_df["user_id_num"].iloc[0]),
        }


def build_weekpart_identity_map(activity_chains: list[pd.DataFrame]) -> dict[tuple[int, int], list[int]]:
    identity_to_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, act_df in enumerate(activity_chains):
        uid_num = int(act_df["user_id_num"].iloc[0])
        is_weekend = int(act_df["dow"].iloc[0] >= 5)
        identity_to_indices[(uid_num, is_weekend)].append(idx)
    return identity_to_indices


class HybridPKSampler(Sampler[list[int]]):
    def __init__(self, user_to_indices: dict[Any, list[int]], batch_size: int = 64, single_user_ratio: float = 0.5):
        self.user_to_indices = user_to_indices
        self.batch_size = batch_size
        self.multi_day_users = []
        self.single_day_users = []

        for uid_num, indices in user_to_indices.items():
            if len(indices) > 1:
                self.multi_day_users.append(uid_num)
            else:
                self.single_day_users.append(uid_num)

        total_identities = batch_size // 2
        self.n_single = min(int(total_identities * single_user_ratio), len(self.single_day_users))
        self.n_multi = min(total_identities - self.n_single, len(self.multi_day_users))

        batches_multi = len(self.multi_day_users) // self.n_multi if self.n_multi > 0 else 0
        batches_single = len(self.single_day_users) // self.n_single if self.n_single > 0 else 0
        self.num_batches = max(batches_multi, batches_single, 1)

    def __iter__(self):
        multi_pool = self.multi_day_users[:]
        single_pool = self.single_day_users[:]
        random.shuffle(multi_pool)
        random.shuffle(single_pool)

        multi_ptr = 0
        single_ptr = 0
        for _ in range(self.num_batches):
            batch_indices = []

            selected_multi = multi_pool[multi_ptr : multi_ptr + self.n_multi]
            for uid_num in selected_multi:
                batch_indices.extend(random.sample(self.user_to_indices[uid_num], 2))
            multi_ptr += self.n_multi
            if multi_ptr >= len(multi_pool):
                multi_ptr = 0

            selected_single = single_pool[single_ptr : single_ptr + self.n_single]
            for uid_num in selected_single:
                day_idx = self.user_to_indices[uid_num][0]
                batch_indices.extend([day_idx, day_idx])
            single_ptr += self.n_single
            if single_ptr >= len(single_pool):
                single_ptr = 0

            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self.num_batches


def ace_collate_fn(
    batch: list[dict[str, Any]],
    mam_mask_prob: float = 0.15,
    min_mam_masks_per_sequence: int = 0,
) -> dict[str, Any]:
    if not 0.0 <= mam_mask_prob <= 1.0:
        raise ValueError(f"mam_mask_prob must be in [0, 1], got {mam_mask_prob}")
    if min_mam_masks_per_sequence < 0:
        raise ValueError(f"min_mam_masks_per_sequence must be non-negative, got {min_mam_masks_per_sequence}")

    loc_embeddings = [item["loc_embeddings"] for item in batch]
    uids = torch.tensor([item["user_id_num"] for item in batch])
    dows = torch.tensor([item["dow"] for item in batch])
    is_weekend = (dows >= 5).long()
    contrastive_labels = uids.long() * 2 + is_weekend

    start_hour_indices = [item["start_hour_indices"] + 1 for item in batch]
    start_minute_indices = [item["start_minute_indices"] + 1 for item in batch]
    dur_indices = [item["dur_indices"] + 1 for item in batch]

    loc_embeddings_padded = pad_sequence(loc_embeddings, batch_first=True, padding_value=0.0)
    hour_indices_padded = pad_sequence(start_hour_indices, batch_first=True, padding_value=0)
    minute_indices_padded = pad_sequence(start_minute_indices, batch_first=True, padding_value=0)
    dur_indices_padded = pad_sequence(dur_indices, batch_first=True, padding_value=0)

    lengths = torch.tensor([len(x) for x in start_hour_indices])
    max_len = hour_indices_padded.size(1)
    range_matrix = torch.arange(max_len).unsqueeze(0).expand(len(batch), max_len)
    padding_mask = range_matrix >= lengths.unsqueeze(1)

    rand_matrix = torch.rand(hour_indices_padded.shape)
    mam_mask = (rand_matrix < mam_mask_prob) & (~padding_mask)

    if min_mam_masks_per_sequence > 0:
        real_positions = ~padding_mask
        for row_idx in range(mam_mask.size(0)):
            n_real = int(real_positions[row_idx].sum().item())
            n_current = int(mam_mask[row_idx].sum().item())
            n_needed = min(min_mam_masks_per_sequence, n_real) - n_current
            if n_needed > 0:
                candidates = torch.where(real_positions[row_idx] & (~mam_mask[row_idx]))[0]
                chosen = candidates[torch.randperm(candidates.numel())[:n_needed]]
                mam_mask[row_idx, chosen] = True

    return {
        "loc_embeddings": loc_embeddings_padded,
        "start_hour_indices": hour_indices_padded,
        "start_minute_indices": minute_indices_padded,
        "dur_indices": dur_indices_padded,
        "mam_mask": mam_mask,
        "padding_mask": padding_mask,
        "labels": contrastive_labels,
        "user_ids": uids,
        "is_weekend": is_weekend,
        "dow": dows,
    }


def build_split_from_users(
    activity_chains: list[pd.DataFrame],
    user_to_indices: dict[int, list[int]],
    selected_users: set[int],
) -> tuple[list[pd.DataFrame], dict[int, list[int]]]:
    split_chains: list[pd.DataFrame] = []
    split_user_map: dict[int, list[int]] = defaultdict(list)
    for uid in selected_users:
        for old_idx in user_to_indices[uid]:
            new_idx = len(split_chains)
            split_chains.append(activity_chains[old_idx])
            split_user_map[uid].append(new_idx)
    return split_chains, split_user_map
