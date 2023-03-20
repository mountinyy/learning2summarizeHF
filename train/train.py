from typing import Tuple

import os

import torch
import wandb
from tqdm import tqdm


def collate_fn(batch):
    return batch


def train_model(conf, args):
    # dataset 설정
    train_dataset = None
    valid_dataset = None
    # DataLoader 설정
    train_dataloader = None
    valid_dataloader = None

    # wandb 설정
    wandb.login()
    if conf.wandb.run_name:
        wandb.init(project=conf.wandb.project_name, name=args.model_name + "-" + conf.wandb.run_name)
    else:
        wandb.init(project=conf.wandb.project_name, name=args.model_name)

    # Model 설정 (model, tokenizer, (config))
    model = None
    tokenizer = None

    # Train Parameter 설정
    optimizer = None
    scheduler = None
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-7)
    save_path = os.path.join(
        conf.model.save_path, args.model_name, conf.model.save_name, conf.wandb.run_name if conf.wandb.run_name else ""
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train/valid loop
    best_acc = 0
    for epoch in range(1, conf.common.num_train_epochs + 1):
        # Train
        total_loss = 0
        total_correct = 0
        model.to(device)
        model.train()
        for data in tqdm(train_dataloader, desc=f"train {epoch} epochs"):
            contexts, labels = extract_data(data, tokenizer, device)
            outputs, correct = run_model("train", model, contexts, labels)
            total_correct += correct

            optimizer.zero_grad()
            outputs["loss"].backward()
            optimizer.step()
            total_loss += outputs["loss"].detach().cpu()

        log_metric(
            "train",
            {
                "total_loss": total_loss,
                "len_dataloader": len(train_dataloader),
                "total_correct": total_correct,
                "len_dataset": len(train_dataset),
            },
        )

        # Valid
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
