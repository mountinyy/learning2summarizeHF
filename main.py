import argparse
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from train.actor_train import train_actor
from train.critic_train import train_reward
from train.datasets import AnthropicDataset


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--run",
        required=True,
        choices=["sft", "rm", "rl", "all"],
        help="Choose what kind of model to train",
    )

    return arg_parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    args = get_args()
    conf = OmegaConf.load("./config.yaml")
    set_seed(conf.common.seed)

    # Download Dataset
    download_dataset = AnthropicDataset(conf)
    download_dataset.save_dataset()

    if args.run == "sft":
        print("=" * 50 + "\nTraining SFT\n" + "=" * 50)
        train_actor(conf)
    elif args.run == "rm":
        print("=" * 50 + "\nTraining RM\n" + "=" * 50)
        train_reward(conf)
    elif args.run == "rl":
        print("=" * 50 + "\nTraining RL\n" + "=" * 50)
        pass
    elif args.run == "all":
        print("=" * 50 + "\nTraining ALL\n" + "=" * 50)
        pass
