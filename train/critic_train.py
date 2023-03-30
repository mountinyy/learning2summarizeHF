import os

import torch
from torch.nn.functional import logsigmoid
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from train.datasets import RewardDataset
from train.models import Critic


# TODO RM 모델 훈련에 score가 필요없다고 판단되면 REwardDataset에서 score 뱉는거 지우기
def collate_fn(batch):
    win_data = []
    lost_data = []
    win_data.append([item[0][0] + item[1][0] for item in batch])
    # win_data.append([item[1][0] for item in batch])
    win_data.append([item[2][0] for item in batch])
    lost_data.append([item[0][1] + item[1][1] for item in batch])
    # lost_data.append([item[1][1] for item in batch])
    lost_data.append([item[2][1] for item in batch])

    return (win_data, lost_data)


def train_reward(conf):
    # dataset 설정
    train_dataset = RewardDataset(os.path.join(conf.dataset.save_path, conf.dataset.rm_path), conf.common.data_limit)
    valid_dataset = RewardDataset(
        os.path.join(conf.dataset.save_path, conf.dataset.rm_path), conf.common.data_limit, is_valid=True
    )
    # DataLoader 설정
    train_dataloader = DataLoader(train_dataset, batch_size=conf.common.batch_size, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=conf.common.batch_size, collate_fn=collate_fn)

    # wandb 설정
    if conf.wandb.use:
        wandb.login()
        if conf.wandb.run_name:
            wandb.init(project=conf.wandb.project_name, name=conf.rm.model_name + "-" + conf.wandb.run_name)
        else:
            wandb.init(project=conf.wandb.project_name, name=conf.rm.model_name)

    # Model 설정 (model, tokenizer, (config))
    model = Critic(conf, is_reward=True)

    # Train Parameter 설정
    optimizer = AdamW(model.parameters(), lr=conf.rm.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_dataset.len // conf.common.batch_size if train_dataset.len // conf.common.batch_size > 0 else 1,
        T_mult=1,
        eta_min=conf.rm.learning_rate * 0.01,
    )
    # loss_fn = MSELoss()

    # model & checkpoint path
    save_path = os.path.join(
        conf.model.save_path,
        "rm",
        conf.rm.model_name,
        conf.wandb.run_name if conf.wandb.run_name else "",
        "model_finished",
    )
    checkpoint_path = os.path.join(
        conf.model.save_path,
        "rm",
        conf.rm.model_name,
        conf.wandb.run_name if conf.wandb.run_name else "",
        "checkpoint",
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load checkpoint
    start_epoch = 1
    if conf.model.checkpoint_name != "None":
        model, optimizer, start_epoch = load_checkpoint(
            model, optimizer, checkpoint_path, conf.model.checkpoint_name, device=device
        )
        print(f"checkpoint {conf.model.checkpoint_name} loaded.")

    # train/valid loop
    for epoch in range(start_epoch, conf.common.num_train_epochs + 1):
        # Train
        total_loss = 0
        model.to(device)
        model.train()
        pbar = tqdm(
            enumerate(train_dataloader),
            total=int(train_dataset.len / conf.common.batch_size),
            desc=f"train {epoch} epochs",
        )
        model.zero_grad()
        # win_data : (datatype, batch, item)
        for i, (win_data, lost_data) in pbar:
            win_rewards, lost_rewards = [], []
            # calculate win_rewards
            tokenized_prompts = model.tokenizer(win_data[0], padding=True, truncation=True, return_tensors="pt")
            actions = tokenized_prompts["input_ids"].to(device)
            actions_mask = tokenized_prompts["attention_mask"].to(device)
            win_rewards = model(actions, actions_mask)

            tokenized_prompts = model.tokenizer(lost_data[0], padding=True, truncation=True, return_tensors="pt")
            actions = tokenized_prompts["input_ids"].to(device)
            actions_mask = tokenized_prompts["attention_mask"].to(device)
            lost_rewards = model(actions, actions_mask)

            # compute loss
            loss = -1 * logsigmoid(win_rewards - lost_rewards).mean()

            loss = loss / conf.rm.gradient_accumulation
            loss.backward()
            loss_value = loss.item()
            total_loss += loss_value
            if conf.wandb.use:
                wandb.log({"train/loss": loss_value})
            pbar.set_postfix_str(f"loss {loss_value}")
            if (i + 1) % conf.rm.gradient_accumulation == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        mean_loss = total_loss / (train_dataset.len / conf.common.batch_size)
        print(f"mean_loss : {mean_loss}")
        if conf.wandb.use:
            wandb.log({"train/mean_loss": mean_loss})

        # Valid
        torch.cuda.empty_cache()
        model.eval()
        pbar = tqdm(
            enumerate(valid_dataloader),
            total=int(valid_dataset.len / conf.common.batch_size),
            desc=f"valid {epoch} epochs",
        )
        total_loss = 0
        with torch.no_grad():
            for i, (win_data, lost_data) in pbar:
                win_rewards, lost_rewards = [], []

                # calculate win_rewards
                for prompts, scores in zip(win_data[0], win_data[1]):
                    tokenized_prompts = model.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
                    actions = tokenized_prompts["input_ids"].to(device)
                    actions_mask = tokenized_prompts["attention_mask"].to(device)
                    reward = model(actions, actions_mask)
                    win_rewards.append(reward)
                win_rewards = torch.Tensor(win_rewards)

                # calculate lost_rewards
                for prompts, scores in zip(lost_data[0], lost_data[1]):
                    tokenized_prompts = model.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
                    actions = tokenized_prompts["input_ids"].to(device)
                    actions_mask = tokenized_prompts["attention_mask"].to(device)
                    reward = model(actions, actions_mask)
                    lost_rewards.append(reward)
                lost_rewards = torch.Tensor(lost_rewards)

                # compute loss
                loss = binary_loss(win_rewards, lost_rewards)
                loss_value = loss.item()
                total_loss += loss_value
                if conf.wandb.use:
                    wandb.log({"valid/loss": loss_value})
                pbar.set_postfix_str(f"loss {loss_value}")

        mean_loss = total_loss / (valid_dataset.len / conf.common.batch_size)
        if conf.wandb.use:
            wandb.log({"valid/mean_loss": mean_loss})

        # save checkpoint
        if epoch % conf.common.checkpoint_epoch == 0:
            path = os.path.join(checkpoint_path, f"{epoch}_checkpoint.pt")
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)
    torch.save({"state_dict": model.state_dict()}, os.path.join(save_path, "model.pt"))


def load_checkpoint(model, optimizer, checkpoint_path, name, device):
    path = os.path.join(checkpoint_path, name)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["state_dict"])
    epoch = ckpt["epoch"]

    # optimizer.load_state_dict(ckpt["optimizer"], map_location=torch.device('cpu'))
    optimizer.load_state_dict(ckpt["optimizer"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return model, optimizer, epoch


def binary_loss(win, lost):
    loss = -logsigmoid(win - lost).mean()
    return loss


class LogSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, win, lost):
        loss = -logsigmoid(win * lost).mean()
        return loss
