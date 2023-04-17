import random

import torch

from utils.experience import Experience


class Memory:
    def __init__(self, batch_size, limit, device):
        self.batch_size = batch_size  # total sampling batch size will be batch_size * experience_batch_size
        self.limit = limit
        self.buffer = []
        self.device = device

    def append(self, experience: Experience):
        experience.to_device(torch.device("cpu"))
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def clear(self):
        self.buffer.clear()

    def sample(self):
        experiences = [experience.to_device(self.device) for experience in random.sample(self.buffer, self.batch_size)]
        return self.collate_fn(experiences)

    def collate_fn(self, experiences):
        device = experiences[0].reward.device
        batch = {
            "reward": torch.concat([item.reward for item in experiences], dim=0).to(device),
            "old_action_prob": torch.concat([item.old_action_prob for item in experiences], dim=0).to(device),
            "action_attention_mask": torch.concat([item.action_attention_mask for item in experiences], dim=0).to(
                device
            ),
            "old_value": torch.concat([item.old_value for item in experiences], dim=0).to(device),
            "actions": torch.concat([item.actions for item in experiences], dim=0).to(device),
            "advantage": torch.concat([item.advantage for item in experiences], dim=0).to(device),
        }
        return batch
