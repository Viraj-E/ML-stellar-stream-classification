
"""Losses for the isolated LitePT astronomy experiment."""

from typing import List

import torch
import torch.nn.functional as F


def _inner_prod(prob: torch.Tensor, label: torch.Tensor, axis: List[int]):
    return (prob * label).sum(dim=axis, keepdim=True)


class _Tanimoto(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p: torch.Tensor, l: torch.Tensor, d: int, axis: List[int]):
        pl = _inner_prod(p, l, axis)
        pp = _inner_prod(p, p, axis)
        ll = _inner_prod(l, l, axis)
        a = 2 ** d
        b = -(2.0 * a - 1.0)
        den = a * (pp + ll) + b * pl
        scale = torch.reciprocal(den)
        scale = torch.nan_to_num(scale, nan=0.0, posinf=1.0, neginf=-1.0)
        ctx.save_for_backward(p, l, pl, pp, ll, scale)
        ctx.a = a
        return pl * scale

    @staticmethod
    def backward(ctx, grad_output):
        p, l, pl, pp, ll, scale = ctx.saved_tensors
        a = ctx.a
        ascale2 = (a * scale) * scale
        pp_plus_ll = pp + ll
        result_p = ascale2 * (-2.0 * p * pl + l * pp_plus_ll)
        result_l = ascale2 * (-2.0 * l * pl + p * pp_plus_ll)
        return result_p * grad_output, result_l * grad_output, None, None


class FTanimoto(torch.nn.Module):
    """Fractal Tanimoto set similarity with complement, depth=0 by default."""

    def __init__(self, depth: int = 0, axis=None):
        super().__init__()
        self.depth = int(depth)
        self.axis = [0] if axis is None else list(axis)
        self.scale = 1.0 if self.depth == 0 else 1.0 / (self.depth + 1.0)

    def _base(self, preds: torch.Tensor, labels: torch.Tensor):
        if self.depth == 0:
            return _Tanimoto.apply(preds, labels, self.depth, self.axis)
        out = 0.0
        for d in range(self.depth + 1):
            out = out + _Tanimoto.apply(preds, labels, d, self.axis)
        return out * self.scale

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        sim = self._base(preds, labels)
        sim = sim + self._base(1.0 - preds, 1.0 - labels)
        return 0.5 * sim


class FTNMTPointLoss(torch.nn.Module):
    """FTNMT loss over the variable-size point/voxel set in a LitePT batch."""

    def __init__(self, depth: int = 0, num_classes: int = 2):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ftnmt = FTanimoto(depth=depth, axis=[0])

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        labels = F.one_hot(target.long(), num_classes=self.num_classes).to(dtype=probs.dtype)
        return (1.0 - self.ftnmt(probs, labels)).mean()


class CrossEntropyPointLoss(torch.nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        return F.cross_entropy(logits, target.long())


class BatchBalancedCrossEntropyPointLoss(torch.nn.Module):
    """Cross entropy with per-batch inverse-frequency class weights."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        target = target.long()
        counts = torch.bincount(target, minlength=logits.shape[1]).to(dtype=logits.dtype, device=logits.device)
        counts = counts.clamp_min(1.0)
        weights = counts.sum() / (counts * float(logits.shape[1]))
        weights = weights / weights.mean().clamp_min(1e-8)
        return F.cross_entropy(logits, target, weight=weights)


class HybridCEFTNMTPointLoss(torch.nn.Module):
    """Weighted CE + FTNMT. CE improves calibration; FTNMT preserves set overlap pressure."""

    def __init__(self, depth: int = 0, num_classes: int = 2, ce_weight: float = 0.5, ftnmt_weight: float = 0.5):
        super().__init__()
        self.ce = CrossEntropyPointLoss()
        self.ftnmt = FTNMTPointLoss(depth=depth, num_classes=num_classes)
        self.ce_weight = float(ce_weight)
        self.ftnmt_weight = float(ftnmt_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        return self.ce_weight * self.ce(logits, target) + self.ftnmt_weight * self.ftnmt(logits, target)
# Portions derived from LitePT: https://github.com/prs-eth/LitePT
# Original copyright (c) 2025 Photogrammetry and Remote Sensing Lab.
# LitePT and these modifications are distributed under the MIT License.
# See ../licenses/LitePT-LICENSE and ../THIRD_PARTY_NOTICES.md.
