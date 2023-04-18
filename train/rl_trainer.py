import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from train.datasets import RLDataset
from train.models import Actor, Critic
from utils.computation import log_prob
from utils.experience import ExperienceController
from utils.loss import CriticLoss, PolicyLoss
from utils.memory import Memory


# TODO RM 모델 훈련에 score가 필요없다고 판단되면 REwardDataset에서 score 뱉는거 지우기
def collate_fn(batch):
    return batch


class RLTrainer:
    def __init__(self, conf):
        self.conf = conf
        self.save_path = os.path.join(
            conf.model.save_path,
            "rl",
            conf.rl.actor_model_name,
            conf.wandb.run_name if conf.wandb.run_name else "",
            "model_finished",
        )
        self.checkpoint_path = os.path.join(
            conf.model.save_path,
            "rl",
            conf.rl.actor_model_name,
            conf.wandb.run_name if conf.wandb.run_name else "",
            "checkpoint",
        )
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.best_loss = 9e5

    def train(self):
        # dataset 설정
        train_dataset = RLDataset(
            os.path.join(self.conf.dataset.save_path, self.conf.dataset.rl_path), self.conf.common.data_limit
        )
        valid_dataset = RLDataset(
            os.path.join(self.conf.dataset.save_path, self.conf.dataset.rl_path),
            self.conf.common.data_limit,
            is_valid=True,
        )
        # DataLoader 설정
        train_dataloader = DataLoader(train_dataset, batch_size=self.conf.common.batch_size, collate_fn=collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.conf.common.batch_size, collate_fn=collate_fn)
        valid_dataloader

        # WanDB 설정
        if self.conf.wandb.use:
            wandb.login()
            if self.conf.wandb.run_name:
                wandb.init(
                    project=self.conf.wandb.project_name,
                    name=f"Actor {self.conf.rl.actor_model_name} Critic {self.conf.rl.critic_model_name}"
                    + "-"
                    + self.conf.wandb.run_name,
                )
            else:
                wandb.init(
                    project=self.conf.wandb.project_name,
                    name=f"Actor {self.conf.rl.actor_model_name} Critic {self.conf.rl.critic_model_name}",
                )

        # Model 설정
        self.sft = Actor(self.conf)
        self.sft.load_state_dict(torch.load(self.conf.rl.actor_saved_path)["state_dict"])
        print("SFT loaded")
        self.critic = Critic(self.conf, is_reward=False)
        self.critic.load_state_dict(torch.load(self.conf.rl.critic_saved_path)["state_dict"])
        print("critic loaded")
        self.rl = Actor(self.conf)
        self.rl.load_state_dict(torch.load(self.conf.rl.actor_saved_path)["state_dict"])
        print("rl loaded")
        self.rm = Critic(self.conf, is_reward=True)
        self.rm.load_state_dict(torch.load(self.conf.rl.critic_saved_path)["state_dict"])
        print("rm loaded")

        # Train Parameter 설정
        self.rl_optimizer = AdamW(self.rl.parameters(), lr=self.conf.rl.actor_learning_rate, weight_decay=1e-5)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=self.conf.rl.critic_learning_rate, weight_decay=1e-5)
        self.rl_scheduler = CosineAnnealingWarmRestarts(
            self.rl_optimizer,
            # T_0=int(self.conf.rl.episodes / 10 * 2),
            T_0=2,
            T_mult=2,
            eta_min=self.conf.rm.learning_rate * 0.001,
        )
        self.critic_scheduler = CosineAnnealingWarmRestarts(
            self.critic_optimizer,
            # T_0=int(self.conf.rl.episodes / 10 * 2),
            T_0=2,
            T_mult=2,
            eta_min=self.conf.rm.learning_rate * 0.001,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.rl.to(device)
        self.critic.to(device)

        # Loss 설정
        self.rl_loss_fn = PolicyLoss(self.conf.rl.ppo_clip_ratio)
        self.critic_loss_fn = CriticLoss(self.conf.rl.value_clip_ratio)

        self.experience_ctr = ExperienceController(
            self.sft,
            self.rm,
            self.rl,
            self.critic,
            self.conf.rl.kl_coef,
        )
        time = 0
        self.memory = Memory(self.conf.common.batch_size, -1, device)
        for episode in range(1, self.conf.rl.episodes + 1):
            for timestep in tqdm(
                range(1, self.conf.rl.max_timestep + 1), desc=f"Episode {episode}/{self.conf.rl.episodes}"
            ):
                time += 1
                data = next(iter(train_dataloader))
                self.experience_ctr.sft.to(device)
                self.experience_ctr.rm.to(device)
                experience = self.experience_ctr.create_experience(data, device)
                self.memory.append(experience)

                if time % self.conf.rl.update_timestep == 0:
                    self.experience_ctr.sft.to("cpu")
                    self.experience_ctr.rm.to("cpu")
                    self.learn()
                    self.memory.clear()
            if episode % self.conf.common.checkpoint_epoch == 0:
                self.save_checkpoint(episode)

            self.rl_scheduler.step()
            self.critic_scheduler.step()
        self.save()
        print(f"checkpoints saved at {self.checkpoint_path}")
        print(f"model saved at {self.save_path}")

    def learn(self):
        for _ in tqdm(range(self.conf.rl.max_epoch), desc="Learn proceduer"):
            experiences = self.memory.sample()
            self.rl_train(experiences)

    def rl_train(self, experience):
        self.rl.train()
        self.critic.train()

        # RL train
        self.rl_optimizer.zero_grad()
        old_action_probs = experience["old_action_prob"]
        total_actions = experience["actions"]
        actions_attention_mask = experience["action_attention_mask"]
        advantages = experience["advantage"]
        action_probs = self.rl(total_actions, actions_attention_mask)
        action_probs = log_prob(action_probs, total_actions)

        rl_loss = self.rl_loss_fn(action_probs, old_action_probs, advantages)
        rl_loss.backward()
        self.rl_optimizer.step()
        rl_loss = rl_loss.item()
        # ptx train 생략
        # https://github.com/hpcaitech/ColossalAI/blob/1a809eddaa20d617e41439c9f4721c05d16a777a/applications/Chat/coati/trainer/ppo.py#L97-L104

        # Critic train
        self.critic_optimizer.zero_grad()
        rewards = experience["reward"]
        old_value = experience["old_value"]
        critic_actions = experience["critic_action"]
        value = self.critic(critic_actions)
        critic_loss = self.critic_loss_fn(value, old_value, rewards)
        critic_loss.backward()
        self.critic_optimizer.step()
        critic_loss = critic_loss.item()
        if self.conf.wandb.use:
            wandb.log(
                {
                    "RL/loss": rl_loss,
                    "RL/learning_rate": self.rl_optimizer.param_groups[0]["lr"],
                    "Critic/loss": critic_loss,
                    "Critic/learning_rate": self.critic_optimizer.param_groups[0]["lr"],
                }
            )

        # save best model
        """
        if abs(rl_loss) < self.best_loss:
            self.best_loss = rl_loss
            self.save(text="best_")
        """

    def save_checkpoint(self, episode):
        path = os.path.join(self.checkpoint_path, f"{episode}_checkpoint.pt")
        torch.save(
            {"state_dict": self.rl.state_dict(), "optimizer": self.rl_optimizer.state_dict(), "epoch": episode}, path
        )

    def save(self, text=""):
        torch.save({"state_dict": self.rl.state_dict()}, os.path.join(self.save_path, text + "model.pt"))
