import torch
import torch.nn.functional as F

def identify_axis(x: torch.Tensor):
    """
    Return the spatial axes for summation in a 4D or 5D tensor.
    4D: (N, C, H, W) -> axes = (2,3)
    5D: (N, C, D, H, W) -> axes = (2,3,4)
    """
    dims = x.dim()
    if dims == 5:
        return (2, 3, 4)
    elif dims == 4:
        return (2, 3)
    else:
        raise ValueError('Tensor must be 4D or 5D.')

def symmetric_unified_focal_multiclass(delta: float = 0.6,
                                       gamma: float = 0.5,
                                       weight: float = 0.5,
                                       smooth: float = 1e-6):
    """
    Symmetric Unified Focal Loss for multiclass segmentation.
    Args:
      delta: trade-off parameter (paper’s δ).
      gamma: focal parameter (paper’s γ).
      weight: λ weighting between focal-CE and focal-Tversky terms.
      smooth: small constant for numeric stability.
    """
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        # y_pred, y_true: (N, C, H, W) or (N, C, D, H, W)
        eps = smooth
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        axes = identify_axis(y_pred)
        
        # 1) Modified focal (CE) component
        ce = -y_true * torch.log(y_pred)                # (N, C, ...)
        mod = (1 - y_pred).pow(1 - gamma)              # (N, C, ...)
        mF = delta * mod * ce                          # (N, C, ...)
        mF = mF.sum(dim=1).sum(dim=axes).mean()        # scalar
        
        # 2) Modified focal Tversky component
        tp = torch.sum(y_true * y_pred, dim=axes)      # (N, C)
        fn = torch.sum(y_true * (1 - y_pred), dim=axes)
        fp = torch.sum((1 - y_true) * y_pred, dim=axes)
        mTI = (tp + eps) / (tp + delta * fn + (1 - delta) * fp + eps)
        mFT = (1 - mTI).pow(gamma)                     # (N, C)
        mFT = mFT.sum(dim=1).mean()                    # scalar
        
        return weight * mF + (1 - weight) * mFT
    
    return loss_fn

def asymmetric_unified_focal_multiclass(rare_classes,
                                        delta: float = 0.6,
                                        gamma: float = 0.5,
                                        weight: float = 0.5,
                                        smooth: float = 1e-6):
    """
    Asymmetric Unified Focal Loss for multiclass segmentation with multiple rare classes.
    Args:
      rare_classes: int or list of ints for rare class indices.
      delta: trade-off parameter (paper’s δ).
      gamma: focal parameter (paper’s γ).
      weight: λ weighting between focal-CE and focal-Tversky terms.
      smooth: small constant for numeric stability.
    """
    if isinstance(rare_classes, int):
        rare_classes = [rare_classes]
    
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        eps = smooth
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        axes = identify_axis(y_pred)
        C = y_pred.shape[1]
        
        # 1) Modified asymmetric focal (CE) component
        ce = -y_true * torch.log(y_pred)               # (N, C, ...)
        # build exponents and weights per class
        exps = torch.full((C,), 1 - gamma, device=y_pred.device)
        wts  = torch.full((C,), 1 - delta, device=y_pred.device)
        for cls in rare_classes:
            exps[cls] = 1.0
            wts[cls]  = delta
        shape = [1, C] + [1] * (y_pred.dim() - 2)
        exps = exps.view(shape)
        wts  = wts.view(shape)
        
        mF = wts * (1 - y_pred).pow(exps) * ce         # (N, C, ...)
        mF = mF.sum(dim=1).sum(dim=axes).mean()        # scalar
        
        # 2) Modified asymmetric focal Tversky component
        tp = torch.sum(y_true * y_pred, dim=axes)      # (N, C)
        fn = torch.sum(y_true * (1 - y_pred), dim=axes)
        fp = torch.sum((1 - y_true) * y_pred, dim=axes)
        mTI = (tp + eps) / (tp + delta * fn + (1 - delta) * fp + eps)
        
        exp_mask = torch.ones_like(mTI)
        for cls in rare_classes:
            exp_mask[:, cls] = 1 - gamma
        
        mFT = (1 - mTI).pow(exp_mask)                 # (N, C)
        mFT = mFT.sum(dim=1).mean()                   # scalar
        
        return weight * mF + (1 - weight) * mFT
    
    return loss_fn