

import torch
import torch.nn as nn
import torch.nn.functional as F




def cross_entropy(
    outs: torch.Tensor, 
    labels: torch.Tensor, 
    reduction: str = "mean"
) -> torch.Tensor:
    return F.cross_entropy(outs, labels, reduction=reduction)

def kl_divergence(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    reduction: str = "batchmean"
) -> torch.Tensor:
    # KL divergence
    assert logits.size() == targets.size()
    # targets = targets.clone().detach()
    inputs = F.log_softmax(logits, dim=-1)
    targets = F.softmax(targets, dim=-1)
    return F.kl_div(inputs, targets, reduction=reduction)
