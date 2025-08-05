import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce_loss(logits, labels, pos_weight=1.0, neg_weight=1.0): 
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

    weights = torch.where(labels == 1, 
                         pos_weight * torch.ones_like(labels),
                         neg_weight * torch.ones_like(labels))
    weighted_loss = loss * weights
    
    return weighted_loss.mean()



def contrastive_loss(pos_scores, neg_scores, temperature=1.0):
    pos_scores = pos_scores / temperature
    neg_scores = neg_scores / temperature

    numerator = pos_scores
    denominator = torch.logsumexp(torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1), dim=1)

    loss = - (numerator - denominator)
    
    return loss.mean()


def dual_contrastive_loss(pos_scores, neg_host_scores, neg_phage_scores, temperature=1.0):
    host_loss = contrastive_loss(pos_scores, neg_host_scores, temperature)
    phage_loss = contrastive_loss(pos_scores, neg_phage_scores, temperature)

    return host_loss + phage_loss


