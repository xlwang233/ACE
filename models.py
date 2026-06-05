"""Activity-chain encoder model for ACE."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TemporalEmbedding(nn.Module):
    def __init__(self, d_input: int, embed_info: str = "all"):
        super().__init__()
        self.embed_info = embed_info
        self.minute_size = 4
        hour_size = 24
        num_dow = 7
        if embed_info == "all":
            self.minute_embed = nn.Embedding(self.minute_size + 1, d_input, padding_idx=0)
            self.hour_embed = nn.Embedding(hour_size + 1, d_input, padding_idx=0)
            self.dow_embed = nn.Embedding(num_dow, d_input)
        elif embed_info == "time":
            self.minute_embed = nn.Embedding(self.minute_size + 1, d_input, padding_idx=0)
            self.hour_embed = nn.Embedding(hour_size + 1, d_input, padding_idx=0)
        elif embed_info == "dow":
            self.dow_embed = nn.Embedding(num_dow, d_input)
        else:
            raise ValueError(f"Unsupported embed_info: {embed_info}")

    def forward(self, hour_indices: torch.Tensor, minute_indices: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        if self.embed_info == "all":
            hour_x = self.hour_embed(hour_indices)
            minute_x = self.minute_embed(minute_indices)
            dow_x = self.dow_embed(dow).unsqueeze(1).expand(-1, hour_x.size(1), -1)
            out = hour_x + minute_x + dow_x
            out[hour_indices == 0] = 0.0
            return out
        if self.embed_info == "time":
            return self.hour_embed(hour_indices) + self.minute_embed(minute_indices)
        return self.dow_embed(dow)


class ActivityChainEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_input = config["d_input"]
        self.nhead = config["nhead"]
        self.num_encoder_layers = config["num_encoder_layers"]
        self.mam_mask_mode = config.get("mam_mask_mode", "location_only")
        self.user_embedding_mode = config.get("user_embedding_mode", "prompt")
        self.use_prompt_token = config.get("use_prompt_token", True)
        loc_hidden_dim = config.get("loc_hidden_dim", 128)
        embed_info = config.get("embed_info", "all")
        dropout = config.get("dropout", 0.1)

        valid_mask_modes = {"location_only", "all"}
        valid_embedding_modes = {"prompt", "mean", "prompt_mean_concat"}
        if self.mam_mask_mode not in valid_mask_modes:
            raise ValueError(f"Unsupported mam_mask_mode: {self.mam_mask_mode}")
        if self.user_embedding_mode not in valid_embedding_modes:
            raise ValueError(f"Unsupported user_embedding_mode: {self.user_embedding_mode}")
        if self.user_embedding_mode in {"prompt", "prompt_mean_concat"} and not self.use_prompt_token:
            raise ValueError(f"user_embedding_mode={self.user_embedding_mode!r} requires use_prompt_token=True")

        self.loc_projection = nn.Linear(loc_hidden_dim, self.d_input)
        self.temporal_embedding = TemporalEmbedding(self.d_input, embed_info=embed_info)
        self.dur_embedding = nn.Embedding((60 * 24 // 5) + 1, self.d_input, padding_idx=0)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_input))
        nn.init.xavier_uniform_(self.mask_token)

        self.prompt_token_embed = nn.Embedding(2, self.d_input) if self.use_prompt_token else None
        self.prompt_mean_projection = (
            nn.Linear(2 * self.d_input, self.d_input) if self.user_embedding_mode == "prompt_mean_concat" else None
        )

        self.pos_encoder = PositionalEncoding(self.d_input)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_input,
            nhead=self.nhead,
            dim_feedforward=config["dim_feedforward"],
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)
        self.mam_head = nn.Linear(self.d_input, loc_hidden_dim)

    def _masked_mean_pool(self, seq_output: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is None:
            valid_mask = torch.ones(seq_output.shape[:2], dtype=seq_output.dtype, device=seq_output.device)
        else:
            valid_mask = (~padding_mask).to(dtype=seq_output.dtype, device=seq_output.device)
        valid_mask = valid_mask.unsqueeze(-1)
        return (seq_output * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        loc_embeddings: torch.Tensor,
        start_hour: torch.Tensor,
        start_minute: torch.Tensor,
        duration: torch.Tensor,
        dow: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        mam_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            loc_embeddings: (B, Seq, 128) - Frozen spatial embeddings
            start_hour:    (B, Seq)      - Bucket indices
            start_minute:  (B, Seq)      - Bucket indices
            duration:      (B, Seq)      - Bucket indices
            dow:           (B, )      - Day of week (0=Mon...6=Sun)
            padding_mask:  (B, Seq)      - True where padding exists
            mam_mask:      (B, Seq)      - True where MAM masking is applied

        Returns:
            user_embedding: (B, Dim)    - For SupCon Loss (The [PROMPT] output)
            reconstruction: (B, Seq, Dim) - Predicted spatial vectors for MAM loss
        """
        batch_size, seq_len, _ = loc_embeddings.shape
        device = loc_embeddings.device

        loc_x = self.loc_projection(loc_embeddings)
        temporal_x = self.temporal_embedding(start_hour, start_minute, dow)
        duration_x = self.dur_embedding(duration)

        if mam_mask is not None:
            mask_token_expanded = self.mask_token.expand(batch_size, seq_len, self.d_input)
            if self.mam_mask_mode == "location_only":
                loc_x = torch.where(mam_mask.unsqueeze(-1), mask_token_expanded, loc_x)
                x = loc_x + temporal_x + duration_x
            else:
                x = loc_x + temporal_x + duration_x
                x = torch.where(mam_mask.unsqueeze(-1), mask_token_expanded, x)
        else:
            x = loc_x + temporal_x + duration_x

        x = self.pos_encoder(x)
        if self.use_prompt_token:
            is_weekend = (dow >= 5).long()
            prompt = self.prompt_token_embed(is_weekend).unsqueeze(1)
            x_combined = torch.cat([prompt, x], dim=1)
            if padding_mask is not None:
                prompt_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
                combined_padding_mask = torch.cat([prompt_mask, padding_mask], dim=1)
            else:
                combined_padding_mask = None
        else:
            x_combined = x
            combined_padding_mask = padding_mask

        output = self.transformer_encoder(x_combined, src_key_padding_mask=combined_padding_mask)
        if self.use_prompt_token:
            prompt_output = output[:, 0, :]
            seq_output = output[:, 1:, :]
        else:
            prompt_output = None
            seq_output = output

        mean_output = self._masked_mean_pool(seq_output, padding_mask)
        if self.user_embedding_mode == "prompt":
            user_embedding = prompt_output
        elif self.user_embedding_mode == "mean":
            user_embedding = mean_output
        else:
            user_embedding = self.prompt_mean_projection(torch.cat([prompt_output, mean_output], dim=1))

        return user_embedding, self.mam_head(seq_output)


class SupConLoss(nn.Module):
    """Supervised contrastive loss over activity-chain embeddings."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0], device=features.device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        logits = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -torch.nan_to_num(mean_log_prob_pos).mean()


class ContrastiveMAMLoss(nn.Module):
    """Contrastive masked activity modelling loss for spatial embeddings."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, reconstruction: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred_vectors = reconstruction[mask]
        target_vectors = ground_truth[mask]
        if pred_vectors.size(0) == 0:
            return reconstruction.sum() * 0.0
        pred_vectors = F.normalize(pred_vectors, dim=1)
        target_vectors = F.normalize(target_vectors, dim=1)
        logits = torch.matmul(pred_vectors, target_vectors.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return self.cross_entropy(logits, labels)
