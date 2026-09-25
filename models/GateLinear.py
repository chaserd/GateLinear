import math

from layers.Embed import PositionalEmbedding,TokenEmbedding
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
# from layers.Embed import TemporalEmbedding
from layers.RevIN import RevIN, FlowAffine, RevIN_FlowBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.RevIN import RevIN

# =========================================================
# Temporal Embedding
# =========================================================
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, freq='h'):
        super().__init__()
        self.hour_embed = nn.Embedding(24, d_model)
        if freq == 't':
            self.minute_embed = nn.Embedding(4, d_model)

    def forward(self, x_mark):
        x_mark = x_mark.long()
        if x_mark.shape[-1] == 1:
            return self.hour_embed(x_mark[:, :, 0])
        hour_x = self.hour_embed(x_mark[:, :, 3])
        if hasattr(self, 'minute_embed'):
            minute_x = self.minute_embed(x_mark[:, :, 4])
            return hour_x + minute_x
        return hour_x

# =========================================================
# Shared Low-Rank Projector
# =========================================================
class SharedLowRankProjector(nn.Module):
    def __init__(self, enc_in, d_model, pred_len, rank=32):
        super().__init__()
        self.rank = rank
        self.pred_len = pred_len
        self.w1 = nn.Parameter(torch.randn(d_model, rank))
        self.w2 = nn.Parameter(torch.randn(rank, pred_len))
        self.channel_scale = nn.Parameter(torch.ones(enc_in, rank))
        self.bias = nn.Parameter(torch.zeros(enc_in, pred_len))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x):
        latent = torch.einsum('bne,ek->bnk', x, self.w1)
        latent = latent * self.channel_scale.unsqueeze(0)
        out = torch.einsum('bnk,ks->bns', latent, self.w2)
        out = out + self.bias.unsqueeze(0)
        return out

#############  多头 ####################

class LowRankGate(nn.Module):

    def __init__(
        self,
        d_model,
        enc_in,
        pred_len,
        gate_rank=16,
        num_heads=8
    ):
        super().__init__()

        self.enc_in = enc_in
        self.pred_len = pred_len
        self.num_heads = num_heads

        self.shared = nn.Sequential(
            nn.Linear(d_model, gate_rank),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # multiple channel gates
        self.channel_gate = nn.Linear(
            gate_rank,
            enc_in * num_heads
        )

        # multiple temporal gates
        self.temporal_gate = nn.Linear(
            gate_rank,
            pred_len * num_heads
        )

        # learnable fusion
        self.head_weight = nn.Parameter(
            torch.ones(num_heads)
        )

    def forward(self, x):

        """
        x: [B,N,E]
        """

        B = x.shape[0]

        g = x.mean(dim=1)

        g = self.shared(g)

        # ===================================
        # channel gate
        # ===================================

        gc = self.channel_gate(g)

        gc = gc.view(
            B,
            self.num_heads,
            self.enc_in
        )

        gc = torch.sigmoid(gc)

        # ===================================
        # temporal gate
        # ===================================

        gt = self.temporal_gate(g)

        gt = gt.view(
            B,
            self.num_heads,
            self.pred_len
        )

        gt = torch.sigmoid(gt)

        # ===================================
        # outer product
        # ===================================

        gate = (
            gc.unsqueeze(-1)
            *
            gt.unsqueeze(-2)
        )

        # [B,H,N,S]

        # ===================================
        # weighted sum
        # ===================================

        w = torch.softmax(
            self.head_weight,
            dim=0
        )

        gate = (
            gate
            *
            w.view(1,-1,1,1)
        ).sum(dim=1)

        return gate



# =========================================================
# Backbone
# =========================================================
class Backbone(nn.Module):
    def __init__(self, seq_len, d_model, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model)
        )

    def forward(self, x):
        return self.net(x)

# 建议增加自适应K选择策略
def adaptive_k(S, E=512, alpha=0.3):
    """根据预测长度自动选择K"""
    k_max = (E * S) / (E + S)
    return int(alpha * k_max)  # 经验取0.3-0.5倍上限

# =========================================================
# 🔥 主模型：支持消融实验版本
# =========================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        # self.rank = getattr(configs, 'rank', 32)
        self.rank = getattr(configs, 'low_rank_k', adaptive_k(configs.pred_len, configs.d_model, alpha=0.3))
        self.use_norm = configs.use_norm

        # ===================== 消融开关 =====================
        self.use_temporal_emb = getattr(configs, 'use_temporal_emb', True)
        self.use_attention = getattr(configs, 'use_attention', True)
        self.use_gate = getattr(configs, 'use_gate', True)
        self.use_shared_branch = getattr(configs, 'use_shared_branch', True)
        self.use_individual_branch = getattr(configs, 'use_individual_branch', True)
        # ====================================================

        # RevIN
        self.revin = RevIN(self.enc_in)

        # Temporal Embedding
        if self.use_temporal_emb:
            self.temporal_embedding = TemporalEmbedding(d_model=self.enc_in, freq=configs.freq)

        # Backbone
        self.backbone = Backbone(seq_len=self.seq_len, d_model=self.d_model)

        # Shared branch
        if self.use_shared_branch:
            self.shared_projector = nn.Linear(self.d_model, self.pred_len)

        # Individual branch
        if self.use_individual_branch:
            self.individual_projector = SharedLowRankProjector(
                enc_in=self.enc_in, d_model=self.d_model, pred_len=self.pred_len, rank=self.rank
            )

        # Gate
        if self.use_gate:
            self.gate = LowRankGate(d_model=self.d_model, enc_in=self.enc_in, pred_len=self.pred_len, gate_rank=16)

        # Attention
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Tanh(),
                nn.Linear(self.enc_in, 1, bias=False),
                nn.Softmax(dim=1)
            )

    def forecast(self, x_enc, x_mark_enc=None):
        # Norm
        if self.use_norm:
            x_enc = self.revin(x_enc, mode='norm')

        # Temporal Embedding
        if self.use_temporal_emb and x_mark_enc is not None:
            t = self.temporal_embedding(x_mark_enc)
            x_enc = x_enc + t

        # [B,L,N] -> [B,N,L]
        x_enc = x_enc.permute(0, 2, 1)

        # Backbone
        enc_out = self.backbone(x_enc)

        # Attention refinement
        if self.use_attention:
            att_score = self.attention(enc_out.permute(0, 2, 1))
            enc_out = enc_out + (enc_out.permute(0, 2, 1) * att_score).permute(0, 2, 1)

        # ==================== 分支前向 ====================
        shared_out = None
        if self.use_shared_branch:
            shared_out = self.shared_projector(enc_out)

        individual_out = None
        if self.use_individual_branch:
            individual_out = self.individual_projector(enc_out)
        # ====================================================

        # ==================== 门控融合 ====================
        if self.use_gate and self.use_shared_branch and self.use_individual_branch:
            gate = self.gate(enc_out)
            out = gate * shared_out + (1 - gate) * individual_out

        elif self.use_shared_branch:
            out = shared_out

        elif self.use_individual_branch:
            out = individual_out

        else:
            raise ValueError("At least one branch must be enabled!")
        # ====================================================

        # [B,N,S] -> [B,S,N]
        out = out.permute(0, 2, 1)

        # Denorm
        if self.use_norm:
            out = self.revin(out, mode='denorm')

        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]
