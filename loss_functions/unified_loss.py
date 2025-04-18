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

################################
#           Dice loss          #
################################
def dice_loss(delta: float = 0.5, smooth: float = 1e-6):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        axes = identify_axis(y_pred)
        tp = torch.sum(y_true * y_pred,   dim=axes)
        fn = torch.sum(y_true * (1 - y_pred), dim=axes)
        fp = torch.sum((1 - y_true) * y_pred, dim=axes)
        dice = (tp + smooth) / (tp + delta*fn + (1-delta)*fp + smooth)
        return torch.mean(1 - dice)
    return loss_fn

################################
#         Tversky loss         #
################################
def tversky_loss(delta: float = 0.7, smooth: float = 1e-6):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        axes = identify_axis(y_pred)
        tp = torch.sum(y_true * y_pred,   dim=axes)
        fn = torch.sum(y_true * (1 - y_pred), dim=axes)
        fp = torch.sum((1 - y_true) * y_pred, dim=axes)
        tversky = (tp + smooth) / (tp + delta*fn + (1-delta)*fp + smooth)
        return torch.mean(1 - tversky)
    return loss_fn

################################
#       Dice coefficient       #
################################
def dice_coefficient(delta: float = 0.5, smooth: float = 1e-6):
    def score_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        axes = identify_axis(y_pred)
        tp = torch.sum(y_true * y_pred,   dim=axes)
        fn = torch.sum(y_true * (1 - y_pred), dim=axes)
        fp = torch.sum((1 - y_true) * y_pred, dim=axes)
        dice = (tp + smooth) / (tp + delta*fn + (1-delta)*fp + smooth)
        return torch.mean(dice)
    return score_fn

################################
#          Combo loss          #
################################
def combo_loss(alpha: float = 0.5, beta: float = 0.5):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        # Dice part
        dice = dice_coefficient()(y_pred, y_true)

        # Cross-entropy part
        eps = 1e-7
        y_pred_clamped = y_pred.clamp(eps, 1 - eps)
        ce = - y_true * torch.log(y_pred_clamped)  # shape N,C,...
        
        if beta is not None:
            # assumes binary (C=2)
            w = torch.tensor([beta, 1-beta],
                             device=y_pred.device,
                             dtype=y_pred.dtype).view(1, -1, *([1]*(y_pred.dim()-2)))
            ce = ce * w
        
        ce = torch.sum(ce, dim=1)    # sum over channel dim
        ce = torch.mean(ce)          # mean over all elements
        
        if alpha is not None:
            return alpha * ce - (1 - alpha) * dice
        else:
            return ce - dice

    return loss_fn

################################
#      Focal Tversky loss      #
################################
def focal_tversky_loss(delta: float = 0.7, gamma: float = 0.75, smooth: float = 1e-6):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        eps = 1e-7
        y_pred = y_pred.clamp(eps, 1 - eps)
        axes = identify_axis(y_pred)
        tp = torch.sum(y_true * y_pred,   dim=axes)
        fn = torch.sum(y_true * (1 - y_pred), dim=axes)
        fp = torch.sum((1 - y_true) * y_pred, dim=axes)
        tversky = (tp + smooth) / (tp + delta*fn + (1-delta)*fp + smooth)
        return torch.mean((1 - tversky) ** gamma)
    return loss_fn

################################
#          Focal loss          #
################################
def focal_loss(alpha: float = None, gamma: float = 2.0):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        eps = 1e-7
        y_pred = y_pred.clamp(eps, 1 - eps)
        ce = - y_true * torch.log(y_pred)
        focal = (1 - y_pred) ** gamma * ce
        if alpha is not None:
            focal = alpha * focal
        focal = torch.sum(focal, dim=1)
        return torch.mean(focal)
    return loss_fn

################################
#       Symmetric Focal loss   #
################################
def symmetric_focal_loss(delta: float = 0.7, gamma: float = 2.0):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        eps = 1e-7
        y_pred = y_pred.clamp(eps, 1 - eps)
        ce = - y_true * torch.log(y_pred)  # shape N,C,H,W
        
        # binary channels-first: channel dim = 1
        back_ce = ((1 - y_pred[:,0,...]) ** gamma) * ce[:,0,...]
        back_ce = (1 - delta) * back_ce

        fore_ce = ((1 - y_pred[:,1,...]) ** gamma) * ce[:,1,...]
        fore_ce = delta * fore_ce

        return torch.mean(back_ce + fore_ce)
    return loss_fn

#################################
# Symmetric Focal Tversky loss  #
#################################
def symmetric_focal_tversky_loss(delta: float = 0.7, gamma: float = 0.75):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        eps = 1e-6
        axes = identify_axis(y_pred)
        tp = torch.sum(y_true * y_pred,   dim=axes)
        fn = torch.sum(y_true * (1 - y_pred), dim=axes)
        fp = torch.sum((1 - y_true) * y_pred, dim=axes)
        dice = (tp + eps) / (tp + delta*fn + (1-delta)*fp + eps)  # shape [N, C]

        back = (1 - dice[:,0]) * torch.pow(1 - dice[:,0], -gamma)
        fore = (1 - dice[:,1]) * torch.pow(1 - dice[:,1], -gamma)
        return torch.mean(torch.stack([back, fore], dim=1))
    return loss_fn

################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta: float = 0.7, gamma: float = 2.0):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        eps = 1e-7
        y_pred = y_pred.clamp(eps, 1 - eps)
        ce = - y_true * torch.log(y_pred)

        back_ce = ((1 - y_pred[:,0,...]) ** gamma) * ce[:,0,...]
        back_ce = (1 - delta) * back_ce

        fore_ce = ce[:,1,...]
        fore_ce = delta * fore_ce

        return torch.mean(back_ce + fore_ce)
    return loss_fn

#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta: float = 0.7, gamma: float = 0.75):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        eps = 1e-6
        axes = identify_axis(y_pred)
        tp = torch.sum(y_true * y_pred,   dim=axes)
        fn = torch.sum(y_true * (1 - y_pred), dim=axes)
        fp = torch.sum((1 - y_true) * y_pred, dim=axes)
        dice = (tp + eps) / (tp + delta*fn + (1-delta)*fp + eps)  # [N,C]

        back = (1 - dice[:,0])
        fore = (1 - dice[:,1]) * torch.pow(1 - dice[:,1], -gamma)
        return torch.mean(torch.stack([back, fore], dim=1))
    return loss_fn

###########################################
#      Symmetric Unified Focal loss       #
###########################################
def sym_unified_focal_loss(weight: float = 0.5, delta: float = 0.6, gamma: float = 0.5):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        ftl = symmetric_focal_tversky_loss(delta, gamma)(y_pred, y_true)
        fl  = symmetric_focal_loss(delta, gamma)(y_pred, y_true)
        return weight * ftl + (1 - weight) * fl
    return loss_fn

###########################################
#      Asymmetric Unified Focal loss      #
###########################################
def asym_unified_focal_loss(weight: float = 0.5, delta: float = 0.6, gamma: float = 0.5):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
        aftl = asymmetric_focal_tversky_loss(delta, gamma)(y_pred, y_true)
        afl  = asymmetric_focal_loss(delta, gamma)(y_pred, y_true)
        return weight * aftl + (1 - weight) * afl
    return loss_fn
