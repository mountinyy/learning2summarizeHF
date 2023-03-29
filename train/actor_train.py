import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from train.datasets import SFTDataset
from train.models import Actor


def collate_fn(batch):
    return batch


def train_actor(conf):
    # dataset 설정
    train_dataset = SFTDataset(os.path.join(conf.dataset.save_path, conf.dataset.sft_path), conf.common.data_limit)
    valid_dataset = SFTDataset(
        os.path.join(conf.dataset.save_path, conf.dataset.sft_path), conf.common.data_limit, is_valid=True
    )

    # DataLoader 설정
    train_dataloader = DataLoader(train_dataset, batch_size=conf.common.batch_size, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=conf.common.batch_size, collate_fn=collate_fn)

    # wandb 설정
    if conf.wandb.use:
        wandb.login()
        if conf.wandb.run_name:
            wandb.init(project=conf.wandb.project_name, name=conf.sft.model_name + "-" + conf.wandb.run_name)
        else:
            wandb.init(project=conf.wandb.project_name, name=conf.sft.model_name)

    # Model 설정 (model, tokenizer, (config))
    model = Actor(conf)

    # Train Parameter 설정
    optimizer = AdamW(model.parameters(), lr=conf.sft.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=train_dataset.len // conf.common.batch_size, T_mult=1, eta_min=conf.sft.learning_rate * 0.01
    )
    loss_fn = CrossEntropyLoss()
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-7)

    # model & checkpoint path
    save_path = os.path.join(
        conf.model.save_path,
        "sft",
        conf.sft.model_name,
        conf.wandb.run_name if conf.wandb.run_name else "",
        "model_finished",
    )
    checkpoint_path = os.path.join(
        conf.model.save_path,
        "sft",
        conf.sft.model_name,
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
        for i, data in pbar:
            tokenized_data = model.tokenizer(data, padding=True, truncation=True, return_tensors="pt")
            states = tokenized_data["input_ids"]
            states_mask = tokenized_data["attention_mask"]

            input_states = states[:, :-1].to(device)
            input_states_mask = states_mask[:, :-1].to(device)
            output_states = states[:, 1:].to(device)

            total_action_probs = model(input_states, input_states_mask)
            loss = loss_fn(
                total_action_probs.view(output_states.size(0) * output_states.size(1), -1), output_states.view(-1)
            )
            loss = loss / conf.sft.gradient_accumulation
            loss.backward()

            loss_value = loss.item()
            total_loss += loss_value
            if conf.wandb.use:
                wandb.log({"train/loss": loss_value})
            pbar.set_postfix_str(f"loss: {loss_value}")
            if (i + 1) % conf.sft.gradient_accumulation == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        run_validation(conf, model, valid_dataset, valid_dataloader, loss_fn, epoch, device)

        mean_loss = total_loss / (train_dataset.len / conf.common.batch_size)
        if conf.wandb.use:
            wandb.log({"train/mean_loss": mean_loss})

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


def run_validation(conf, model, valid_dataset, valid_dataloader, loss_fn, epoch, device):
    torch.cuda.empty_cache()
    model.eval()
    pbar = tqdm(
        enumerate(valid_dataloader),
        total=int(valid_dataset.len / conf.common.batch_size),
        desc=f"valid {epoch} epochs",
    )
    total_loss = 0
    with torch.no_grad():
        for i, data in pbar:
            tokenized_data = model.tokenizer(data, padding=True, truncation=True, return_tensors="pt")
            states = tokenized_data["input_ids"]
            states_mask = tokenized_data["attention_mask"]

            input_states = states[:, :-1].to(device)
            input_states_mask = states_mask[:, :-1].to(device)
            output_states = states[:, 1:].to(device)

            total_action_probs = model(input_states, input_states_mask)
            loss = loss_fn(
                total_action_probs.view(output_states.size(0) * output_states.size(1), -1), output_states.view(-1)
            )

            loss_value = loss.item()
            total_loss += loss_value
            if conf.wandb.use:
                wandb.log({"valid/loss": loss_value})
            pbar.set_postfix_str(f"loss {loss_value}")

    mean_loss = total_loss / (valid_dataset.len / conf.common.batch_size)
    if conf.wandb.use:
        wandb.log({"valid/mean_loss": mean_loss})
