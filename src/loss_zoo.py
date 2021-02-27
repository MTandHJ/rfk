

import torch
import torch.nn as nn
import torch.nn.functional as F




def cross_entropy(outs, labels, reduction="mean"):
    return F.cross_entropy(outs, labels, reduction=reduction)

def kl_divergence(logits, targets, reduction="batchmean"):
    # KL divergence
    assert logits.size() == targets.size()
    # targets = targets.clone().detach()
    inputs = F.log_softmax(logits, dim=-1)
    targets = F.softmax(targets, dim=-1)
    return F.kl_div(inputs, targets, reduction=reduction)

def contrastive_loss(outs: torch.Tensor, reduction="mean"):
    n = outs.size(0) // 2
    logits = (outs @ outs.T).fill_diagonal_(-1e10)
    labels = torch.arange(n, 3*n, dtype=torch.long, device=logits.device) % (2 * n)
    return cross_entropy(logits, labels, reduction=reduction)


