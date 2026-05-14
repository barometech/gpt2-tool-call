"""LONG-CONTEXT для GPT-2 124M + steering adapter.

Стратегия (CPU-friendly):
  1. Position Interpolation (PI): extend wpe (1024,768) → (N_ctx, 768) bilinear.
     Никакого обучения. Маппинг pos[i] = i * 1024/N_ctx.
  2. Chunked sliding window: prompt разбивается на чанки по CHUNK_SIZE (default 512).
     Каждый чанк форвардится через GPT-2 + adapter, pooled state агрегируется.
  3. Memory token (RMT-style): aggregated pooled state из предыдущих чанков
     добавляется в начало каждого нового чанка как "контекст".
  4. Финальный чанк → vocab_logits с учётом всей истории.

Ссылки:
  - Position Interpolation: arxiv.org/abs/2306.15595
  - RMT: arxiv.org/abs/2304.11062
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, json, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.path.insert(0, "code")

from integrated_gpt2_torch import (
    GPT2, load_gpt2_torch_weights, encode, decode
)
from steering_v2 import FullSteeringGPT2, load_classifier_from_npz

DEVICE = torch.device('cpu')


def interpolate_wpe(model: GPT2, new_max_positions: int):
    """PI: расширить learned wpe (1024, 768) до (new_max_positions, 768) bilinear.

    pos[i] в новой системе мапится на pos[i*1024/new_max_positions] в старой.
    """
    old_wpe = model.wpe.weight.data   # (1024, 768)
    old_n = old_wpe.shape[0]
    new_n = new_max_positions
    # New positions
    old_pos = torch.linspace(0, old_n - 1, new_n)
    floor_pos = torch.floor(old_pos).long().clamp(0, old_n - 1)
    ceil_pos = torch.ceil(old_pos).long().clamp(0, old_n - 1)
    frac = (old_pos - floor_pos.float()).unsqueeze(-1)   # (new_n, 1)
    new_wpe = (1 - frac) * old_wpe[floor_pos] + frac * old_wpe[ceil_pos]
    # Replace
    new_emb = nn.Embedding(new_n, old_wpe.shape[1])
    new_emb.weight.data.copy_(new_wpe)
    model.wpe = new_emb
    print(f"  PI extended wpe: {old_n} → {new_n}  (ratio {old_n/new_n:.2f})")


