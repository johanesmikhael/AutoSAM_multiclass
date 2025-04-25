import torch
import torch.nn as nn
from typing import List
from torch import Tensor

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt



import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_sdf(gt: np.ndarray) -> np.ndarray:
    """Single‐channel signed distance."""
    pos = gt.astype(bool)
    if pos.any():
        neg = ~pos
        dist_out = distance_transform_edt(neg)
        dist_in  = distance_transform_edt(pos)
        return (dist_out - dist_in).astype(np.float32)
    else:
        return np.zeros_like(gt, dtype=np.float32)

def compute_sdf_per_class(one_hot_gt: torch.Tensor) -> torch.Tensor:
    B, C, H, W = one_hot_gt.shape
    sdf = torch.empty((B, C, H, W), dtype=torch.float32)
    for b in range(B):
      for c in range(C):
        sdf[b, c] = torch.from_numpy(compute_sdf(one_hot_gt[b,c].cpu().numpy()))
    return sdf.to(one_hot_gt.device)

class BoundaryLoss(torch.nn.Module):
    def __init__(self, idc: list):
        super().__init__()
        self.idc = idc

    def forward(self, probs: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        # probs: (B,C,H,W), softmaxed; sdf: same shape
        if not torch.allclose(probs.sum(1), 1, atol=1e-4):
            raise ValueError("`probs` must sum to 1 over classes")
        pc = probs[:, self.idc, ...]
        dc = sdf[:,   self.idc, ...]
        # φ positive outside, negative inside
        return (pc * dc).mean()


