import torch
import torch.nn as nn

class DiceScoreCoefficient(nn.Module):
    def __init__(self, n_classes: int, eps: float = 1e-8, ignore_index: int = 0):
        """
        n_classes: number of segmentation labels
        eps: small constant to avoid division by zero
        ignore_index: class index to ignore (e.g. background)
        """
        super().__init__()
        self.n_classes = n_classes
        self.eps = eps
        self.ignore_index = ignore_index
        # confusion_matrix will move with the module (cpu<->gpu)
        self.register_buffer(
            'confusion_matrix',
            torch.zeros((n_classes, n_classes), dtype=torch.int64)
        )

    def reset(self):
        """Zero out the accumulated confusion matrix."""
        self.confusion_matrix.zero_()

    def fast_hist(self, label_true: torch.Tensor, label_pred: torch.Tensor):
        """
        Build a (n_classes × n_classes) histogram, ignoring pixels where
        label_true == ignore_index.
        label_true, label_pred: 1D LongTensor of same shape.
        """
        mask = (
            (label_true >= 0)
            & (label_true < self.n_classes)
            & (label_true != self.ignore_index)
        )
        lt = label_true[mask]
        lp = label_pred[mask]
        combined = lt * self.n_classes + lp
        hist = torch.bincount(
            combined,
            minlength=self.n_classes ** 2
        )
        return hist.view(self.n_classes, self.n_classes)

    def _dsc(self, mat: torch.Tensor):
        """
        Compute per-class Dice = 2·TP / (2·TP + FP + FN).
        Classes with TP=FP=FN=0 are set to NaN (skipped).  The ignored class
        index is always set to NaN as well.
        """
        mat = mat.float()
        tp = torch.diag(mat)
        fp = mat.sum(dim=1)
        fn = mat.sum(dim=0)

        precision = tp / (fp + self.eps)
        recall    = tp / (fn + self.eps)
        dsc = 2 * precision * recall / (precision + recall + self.eps)

        # classes with no ground‐truth pixels
        gt_empty = (tp + fn) == 0
        # among those, did we also predict none?
        pred_empty = (tp + fp) == 0

        # assign 1 or 0 to those cases
        dsc[gt_empty & pred_empty] = float('nan')
        dsc[gt_empty & ~pred_empty] = 0.0

        # still skip the ignore_index entirely
        dsc[self.ignore_index] = float('nan')

        return dsc

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """
        output: (B, n_classes, *spatial_dims) - logits or probabilities
        target: (B, *spatial_dims) with integer class labels

        Returns:
            Tensor of length n_classes containing per-class Dice scores.
            Absent and ignored classes are NaN.
        """
        seg = output.argmax(dim=1)
        lt = target.view(-1)
        lp = seg.view(-1)
        self.confusion_matrix += self.fast_hist(lt, lp)
        return self._dsc(self.confusion_matrix)
