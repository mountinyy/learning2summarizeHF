import random

import torch

from utils.experience import Experience


class Memory:
    def __init__(self, batch_size, limit, device):
        self.batch_size = batch_size
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
        experiences = random.sample(self.buffer, self.batch_size)
        experiences.to_device(self.device)
        return experiences
