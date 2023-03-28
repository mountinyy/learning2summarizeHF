from typing import Tuple

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
    # DataLoader 설정
    train_dataloader = DataLoader(train_dataset, batch_size=conf.common.batch_size, collate_fn=collate_fn)

    # wandb 설정
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
    save_path = os.path.join(
        conf.model.save_path,
        conf.sft.model_name,
        conf.model.save_name,
        conf.wandb.run_name if conf.wandb.run_name else "",
    )
    save_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train/valid loop
    for epoch in range(1, conf.common.num_train_epochs + 1):
        # Train
        total_loss = 0
        model.to(device)
        model.train()
        pbar = tqdm(
            enumerate(train_dataloader),
            total=int(train_dataset.len / conf.common.batch_size),
            desc=f"train {epoch} epochs",
        )
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            total_loss += loss_value
            wandb.log({"train/loss": loss_value})
            pbar.set_postfix_str(f"loss {loss_value}")

        mean_loss = total_loss / (train_dataset.len / conf.common.batch_size)
        wandb.log({"train/mean_loss": mean_loss})

        # Valid
        """
        torch.cuda.empty_cache()
        model.eval()
        total_loss = 0
        total_correct = 0
        with torch.no_grad():
            for data in tqdm(valid_dataloader, desc=f"valid {epoch} epochs"):
                contexts, labels = extract_data(data, tokenizer, device)
                outputs, correct = run_model("valid", model, contexts, labels)

                total_loss += outputs["loss"].detach().cpu()

            log_metric(
                "valid",
                {
                    "total_loss": total_loss,
                    "len_dataloader": len(valid_dataloader),
                    "total_correct": total_correct,
                    "len_dataset": len(valid_dataset),
                },
            )
            wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
            accuracy = total_correct / len(valid_dataset)

            if accuracy > best_acc:
                best_acc = accuracy
                model.save_pretrained(save_path)
            scheduler.step(accuracy)
        """


def extract_data(data, tokenizer, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract data from batch.
    Currently assume batch data is dict type with key 'context', 'label'.
    For context data, tokenized context will be returned.

    Args:
        data (dict): batch data with { 'context': (batch_size, context), 'label': (batch_size, label)}
        tokenizer (transformers.Tokenizer): Tokenizer for tokenized context
        device : device which train is performed.

    Returns:
        Tuple (tokenized_context, labels): (bz, max_len) 크기의 tokenized_context와 labels 리턴
    """
    contexts = [batch["context"] for batch in data]
    labels = torch.LongTensor([batch["label"] for batch in data]).to(device)
    tokenized_contexts = tokenizer(
        contexts, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    return (tokenized_contexts, labels)


def run_model(run_type, model, contexts, labels):
    """Return model outputs and the correct counts compared to labels.

    Args:
        run_type (str): Decide whether to log as 'train' or 'valid'.
        model : model to train
        contexts : contexts for input
        labels : ground truth label.

    Returns:
        tuple of (model outputs, correct_count)
    """
    assert run_type in ["train", "valid"], f"no valid run_type for {run_type}"
    outputs = model(**contexts, labels=labels)
    predicted = outputs["logits"].argmax(dim=-1)
    correct = (predicted == labels).sum().datach().cpu().item()
    wandb.log({f"{run_type}/loss": outputs["loss"]})
    return (outputs, correct)


def log_metric(name, **kwargs):
    """logs metric to wandb.
    User can update for custom metrics.

    Args:
        name (str): name for log. Currently either "train" or "valid"
    """
    mean_loss = kwargs["total_loss"] / kwargs["len_dataloader"]
    accuracy = kwargs["total_correct"] / kwargs["len_dataset"]
    wandb.log(
        {
            f"{name}/mean_loss": mean_loss,
            f"{name}/accuracy": accuracy,
        }
    )
