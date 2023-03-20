import os

from torch.utils.data import Dataset


class TLDRDataset(Dataset):
    def __init__(self, path: str, tokenizer, usage: str):
        data_path = os.path.join(path, f"{usage}.jsonl")
        with open(data_path, "r") as f:
            data = [eval(line.replace("null", "None")) for line in f.readlines()]
        self.contexts = [
            tokenizer(item["post"], padding="max_length", truncation=True, max_length=1024) for item in data
        ]
        self.summaries = [
            tokenizer(item["summary"], padding="max_length", truncation=True, max_length=1024)["input_ids"]
            for item in data
        ]

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        return {"context": self.contexts[idx], "summary": self.summaries[idx]}
