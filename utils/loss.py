import torch
from torch import nn


class PolicyLoss(nn.Module):
    def __init__(self, clip_ratio=0.2):
        super().__init__()
        self.clip_ratio = clip_ratio

    def forward(self, action_probs, old_action_probs, advantages, action_mask=None):
        ratio = (action_probs - old_action_probs).exp()
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * advantages.unsqueeze(-1)
        loss = -torch.min(surr1, surr2)
        # TODO action mask를 추가해서 aciton에 관한 부분만 loss를 계산해야 할까?
        # https://github.com/hpcaitech/ColossalAI/blob/1c7734bc94ac1a7215e08368adc4e7e25e3b8102/applications/Chat/coati/models/loss.py#L43-L44
        return loss.mean()


class CriticLoss(nn.Module):
    def __init__(self, clip_ratio=0.4):
        super().__init__()
        self.clip_ratio = clip_ratio

    def forward(self, values, old_values, reward):
        # TODO 왜 value도 clamp를 하는지?
        values_clipped = old_values + (values - old_values).clamp(-self.clip_ratio, self.clip_ratio)
        surr1 = (values_clipped - reward) ** 2
        surr2 = (values - reward) ** 2
        loss = torch.max(surr1, surr2)
        loss = loss.mean()
        return 0.5 * loss
