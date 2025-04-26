import torch
from kornia.contrib import DistanceTransform


# instantiate once (e.g. at module scope) so the kernel is compiled only once
_DT = DistanceTransform(kernel_size=3, h=0.35)

def compute_sdf_per_class(one_hot_gt: torch.Tensor) -> torch.Tensor:
    """
    Vectorized SDF via Kornia’s DistanceTransform, with per‐class normalization
    and absent‐class → uniform‐penalty (=1) maps.

    Args:
        one_hot_gt: (B, C, H, W) float or bool on CUDA
    Returns:
        sdf_penalty: (B, C, H, W) float, φ>0 outside, φ<0 inside,
                     normalized to [0,1] for present classes and =1 for absent ones.
    """
    B, C, H, W = one_hot_gt.shape
    device     = one_hot_gt.device
    masks      = one_hot_gt.float().to(device)               # (B,C,H,W)

    # flatten batch×class into a single batch for DT
    flat = masks.view(B * C, 1, H, W)
    inv  = 1.0 - flat

    # run two big batched transforms
    _DT.to(device)
    dist_fg = _DT(flat)   # distance‐to‐foreground: zero inside mask
    dist_bg = _DT(inv)    # distance‐to‐background: zero outside mask

    # reshape back and form signed‐distance φ = dist_fg – dist_bg
    sdf = (dist_fg - dist_bg).view(B, C, H, W)

    # per‐class, per‐image normalization into [0,1]
    mins  = sdf.amin(dim=(-2, -1), keepdim=True)             # (B,C,1,1)
    maxs  = sdf.amax(dim=(-2, -1), keepdim=True)             # (B,C,1,1)
    denom = (maxs - mins).clamp(min=1e-6)
    sdf_n = (sdf - mins) / denom                             # (B,C,H,W)

    # build a mask of which classes actually appear
    present = (masks.sum(dim=(-2, -1), keepdim=True) > 0)    # (B,C,1,1)

    # absent classes get a uniform “1.0” penalty map
    ones = torch.ones_like(sdf_n)
    sdf_penalty = torch.where(present, sdf_n, ones)          # (B,C,H,W)

    return sdf_penalty


# Your BoundaryLoss stays the same:
class BoundaryLoss(torch.nn.Module):
    def __init__(self, idc: list):
        super().__init__()
        self.idc = idc

    def forward(self, probs: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        # probs: (B,C,H,W), softmaxed; sdf: same shape
        if not torch.allclose(probs.sum(1), torch.ones_like(probs.sum(1)), atol=1e-4):
            raise ValueError("`probs` must sum to 1 over classes")
        pc = probs[:, self.idc, ...]
        dc = sdf[:,   self.idc, ...]
        return (pc * dc).mean()