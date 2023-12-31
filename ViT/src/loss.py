import torch
import torch.nn.functional as F


class SoftTargetCrossEntropy(torch.nn.Module):
    """Cross Entropy w/ smoothing or soft targets
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/cross_entropy.py
    """

    def __init__(self) -> None:
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
