import torch
import torch.nn as nn
from typing import List
from torch import Tensor

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt



def compute_sdf_per_class(one_hot_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute signed distance maps for one-hot encoded ground truth masks.

    Args:
        one_hot_gt: Tensor of shape (B, C, H, W), one-hot encoded ground truth masks.

    Returns:
        sdf_maps: Tensor of shape (B, C, H, W), signed distance maps for each class.
    """
    B, C, H, W = one_hot_gt.shape
    sdf_maps = torch.zeros_like(one_hot_gt, dtype=torch.float32)

    for b in range(B):
        for c in range(C):
            gt_np = one_hot_gt[b, c].cpu().numpy().astype(np.uint8)
            posmask = gt_np.astype(bool)
            if posmask.any():
                negmask = ~posmask
                dist_out = distance_transform_edt(negmask)
                dist_in = distance_transform_edt(posmask)
                sdf = dist_out - dist_in
            else:
                sdf = np.zeros_like(gt_np, dtype=np.float32)

            sdf_maps[b, c] = torch.from_numpy(sdf)

    return sdf_maps.to(one_hot_gt.device)


def simplex(t: torch.Tensor, axis=1) -> bool:
    _sum = t.sum(dim=axis)
    return torch.allclose(_sum, torch.ones_like(_sum), atol=1e-4)

def one_hot(t: torch.Tensor) -> bool:
    return torch.equal(t, t.round())

class BoundaryLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        # These assertions are used for sanity checks
        assert simplex(probs), "Input probs should be softmax outputs (sum to 1 across classes)."
        assert not one_hot(dist_maps), "Distance maps should not be one-hot encoded."

        # Filter the selected classes (idc)
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        # Element-wise multiplication (no einsum needed)
        multipled = pc * dc

        # Mean over all dimensions
        loss = multipled.mean()

        return loss

# Alias for compatibility
BoundaryLoss = SurfaceLoss
