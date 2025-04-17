import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = 'mean',
    ):
        """
        Focal Loss for multi-class classification, compatible with CrossEntropyLoss.

        Args:
            weight (Tensor, optional): A manual rescaling weight given to each class.
            gamma (float): Focusing parameter for modulating factor (1 - p_t).
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (Tensor): (N, C, ...) logits from the model
            target (Tensor): (N, ...) ground truth class indices
        Returns:
            Tensor: focal loss
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * logpt

        # Apply per-class weighting
        if self.weight is not None:
            weight = self.weight[target]
            loss *= weight

        # Handle ignore_index
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            loss = loss[valid_mask]

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
