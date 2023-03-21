import os

from torch.utils.data import Dataset


class TLDRDataset(Dataset):
    def __init__(self, path: str, tokenizer, usage: str, max_token_length, data_limit=None):
        print(f"initializing {usage} dataset...")
        if data_limit == "None":
            data_limit = None
        data_path = (
            os.path.join(path, "train.jsonl") if usage in ["train", "valid"] else os.path.join(path, "test.jsonl")
        )
        with open(data_path, "r") as f:
            data = [eval(line.replace("null", "None")) for line in f.readlines()[:data_limit]]
        start = int(len(data) * 0.8) if usage == "valid" else None
        end = int(len(data) * 0.8) if usage == "train" else None
        data = data[start:end]

        self.contexts = [
            tokenizer(item["post"], padding="max_length", truncation=True, max_length=max_token_length) for item in data
        ]
        self.summaries = [
            tokenizer(item["summary"], padding="max_length", truncation=True, max_length=max_token_length)["input_ids"]
            for item in data
        ]
        print("done")

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        return {"context": self.contexts[idx], "summary": self.summaries[idx]}
