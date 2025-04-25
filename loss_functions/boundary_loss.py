import torch
from kornia.contrib import DistanceTransform

def compute_sdf_per_class(
    one_hot_gt: torch.Tensor,
    kernel_size: int = 3,
    h: float        = 0.35
) -> torch.Tensor:
    """
    Compute signed “distance” maps using Kornia’s DistanceTransform (L1/Chamfer approx).
    
    Args:
        one_hot_gt: (B, C, H, W) one-hot masks, float or bool on CUDA.
        kernel_size: size of the chamfer kernel (odd integer).
        h:         weighting of diagonal steps (typical 0.35–0.5).
        
    Returns:
        sdf: (B, C, H, W) signed distance maps (positive outside, negative inside).
    """
    B, C, H, W = one_hot_gt.shape
    device     = one_hot_gt.device
    # ensure float for kornia
    masks = one_hot_gt.float().to(device)
    dt    = DistanceTransform(kernel_size=kernel_size, h=h).to(device)

    # preallocate
    sdf = torch.empty((B, C, H, W), dtype=torch.float32, device=device)

    for c in range(C):
        m = masks[:, c:c+1]      # shape (B,1,H,W)
        inv = 1.0 - m            # background mask
        # Manhattan‐approx distance out/in
        dist_out = dt(inv)       # (B,1,H,W)
        dist_in  = dt(m)         # (B,1,H,W)
        sdf[:, c] = (dist_out - dist_in).squeeze(1)

    return sdf


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