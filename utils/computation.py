import torch.nn.functional as F


def log_prob(sequences, labels):
    log_probs = F.log_softmax(sequences, dim=-1)
    log_probs = log_probs[:, :-1, :].gather(dim=-1, index=labels[:, 1:].unsqueeze(-1)).squeeze(-1)
    return log_probs
