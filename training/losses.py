"""Plain binary cross entropy for positive and filtered negative triples."""
import torch
import torch.nn.functional as F


def bce_loss(pos_scores, neg_scores):
    """Mean BCE over all positive and negative examples, without reweighting."""
    scores = torch.cat([pos_scores.reshape(-1), neg_scores.reshape(-1)]).float()
    labels = torch.cat([torch.ones_like(pos_scores).reshape(-1),
                        torch.zeros_like(neg_scores).reshape(-1)]).float()
    return F.binary_cross_entropy_with_logits(scores, labels)
