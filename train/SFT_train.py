from typing import Tuple

import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.functional.text.rouge import rouge_score
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

import wandb
from train.datasets import TLDRDataset


def collate_fn(batch):
    return batch


def train_model(conf, args):
    # Model 설정 (model, tokenizer, (config))
    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = BartTokenizer.from_pretrained(args.model_name)

    # dataset 설정
    train_dataset = TLDRDataset(
        conf.common.dataset_path, tokenizer, "train", conf.common.max_token_length, conf.common.data_limit
    )
    valid_dataset = TLDRDataset(
        conf.common.dataset_path, tokenizer, "valid", conf.common.max_token_length, conf.common.data_limit
    )
    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=conf.common.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=conf.common.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True
    )
    # wandb 설정
    wandb.login()
    if conf.wandb.run_name:
        wandb.init(project=conf.wandb.project_name, name=args.model_name + "-" + conf.wandb.run_name)
    else:
        wandb.init(project=conf.wandb.project_name, name=args.model_name)

    # Train Parameter 설정
    optimizer = AdamW(model.parameters(), lr=conf.common.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-7)
    save_path = os.path.join(
        conf.model.save_path, args.model_name, conf.model.save_name, conf.wandb.run_name if conf.wandb.run_name else ""
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train/valid loop
    best_rouge = 0
    for epoch in range(1, conf.common.num_train_epochs + 1):
        # Train
        total_loss = 0
        rouge_sum = {
            "rouge2_fmeasure": 0,
            "rouge2_precision": 0,
            "rouge2_recall": 0,
        }
        model.to(device)
        model.train()
        for data in tqdm(train_dataloader, desc=f"train {epoch} epochs"):
            contexts, attention_mask, labels = extract_data(data, tokenizer, device)
            outputs, rouge = run_model("train", model, contexts, attention_mask, labels, tokenizer)

            optimizer.zero_grad()
            outputs["loss"].backward()
            optimizer.step()
            total_loss += outputs["loss"].detach().cpu()
            for k in rouge.keys():
                rouge_sum[k] += rouge[k]

        log_metric(
            "train",
            {
                "total_loss": total_loss,
                "len_dataloader": len(train_dataloader),
                "rouge": rouge,
                "len_dataset": len(train_dataset),
            },
        )

        # Valid
        torch.cuda.empty_cache()
        model.eval()
        total_loss = 0
        rouge_sum = {
            "rouge2_fmeasure": 0,
            "rouge2_precision": 0,
            "rouge2_recall": 0,
        }
        with torch.no_grad():
            for data in tqdm(valid_dataloader, desc=f"valid {epoch} epochs"):
                contexts, attention_mask, labels = extract_data(data, tokenizer, device)
                outputs, rouge = run_model("valid", model, contexts, attention_mask, labels, tokenizer)

                total_loss += outputs["loss"].detach().cpu()
                for k in rouge.keys():
                    rouge_sum[k] += rouge[k]

            log_metric(
                "valid",
                {
                    "total_loss": total_loss,
                    "len_dataloader": len(valid_dataloader),
                    "rouge": rouge,
                    "len_dataset": len(valid_dataset),
                },
            )
            wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
            print(f'accuracy {rouge_sum["rouge2_fmeasure"]}, best_rouge {best_rouge}')
            if rouge_sum["rouge2_fmeasure"] > best_rouge:
                print("model saved")
                best_rouge = rouge_sum["rouge2_fmeasure"]
                model.save_pretrained(save_path)
            scheduler.step()


def extract_data(data, tokenizer, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract data from batch.
    Currently assume batch data is dict type with key 'context', 'label'.
    For context data, tokenized context will be returned.

    Args:
        data (dict): batch data with  { 'context': (batch_size, context), 'label': (batch_size, label)}
        tokenizer (transformers.Tokenizer): Tokenizer for tokenized context
        device : device which train is performed.

    Returns:
        Tuple (tokenized_context, labels): (bz, max_len) 크기의 tokenized_context와 labels 리턴
    """
    contexts = torch.LongTensor([batch["context"]["input_ids"] for batch in data]).to(device)
    attention_mask = torch.LongTensor([batch["context"]["attention_mask"] for batch in data]).to(device)
    labels = torch.LongTensor([batch["summary"] for batch in data]).to(device)
    return (contexts, attention_mask, labels)


def run_model(run_type, model, contexts, attention_mask, labels, tokenizer):
    """Return model outputs and the correct counts compared to labels.
    Currently model should get labels as input, and outputs should include loss

    Args:
        run_type (str): Decide whether to log as 'train' or 'valid'.
        model : model to train
        contexts : contexts for input
        labels : ground truth label.

    Returns:
        tuple of (model outputs, correct_count)
    """
    assert run_type in ["train", "valid"], f"no valid run_type for {run_type}"
    outputs = model(contexts, attention_mask=attention_mask, labels=labels)
    predicted = outputs["logits"].argmax(dim=-1)
    rouge = calculate_rouge2(predicted, labels, tokenizer)
    wandb.log({f"{run_type}/loss": outputs["loss"]})
    return (outputs, rouge)


def log_metric(name, values):
    """logs metric to wandb.
    User can update for custom metrics.

    Args:
        name (str): name for log. Currently either "train" or "valid"
    """
    mean_loss = values["total_loss"] / values["len_dataloader"]

    wandb.log(
        {
            f"{name}/mean_loss": mean_loss,
            f"{name}/rouge2_f1": values["rouge"]["rouge2_fmeasure"] / values["len_dataset"],
            f"{name}/rouge2_preciison": values["rouge"]["rouge2_precision"] / values["len_dataset"],
            f"{name}/rouge2_recall": values["rouge"]["rouge2_recall"] / values["len_dataset"],
        }
    )


def calculate_rouge2(predicted, labels, tokenizer):
    d_pred = tokenizer.batch_decode(predicted)
    d_label = tokenizer.batch_decode(labels)
    rouge = rouge_score(d_pred, d_label, rouge_keys="rouge2")
    return rouge
