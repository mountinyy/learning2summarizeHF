import json
import os
import random
from abc import abstractmethod

import numpy as np
from datasets import load_dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm


class BaseDataset(IterableDataset):
    def __init__(self, path, limit, is_valid=False):
        super().__init__()
        self.path = path

        # Fix limit when it is "None"
        limit = None if limit == "None" else limit
        with open(self.path, "r") as f:
            tmp_data = json.load(f)["prompts"][:limit]
        if limit is None:
            limit = len(tmp_data)

        # make boundary for train/valid set to even
        boundary = int(limit * 0.8)
        boundary = boundary if boundary % 2 == 0 else boundary - 1
        if is_valid:
            self.start = boundary
            self.limit = limit
        else:
            self.start = 0
            self.limit = boundary

        self.len = int(len(tmp_data[self.start : self.limit]) / 2)

    @abstractmethod
    def __iter__(self):
        pass


class RLDataset(BaseDataset):
    def __init__(self, path, limit, is_valid=False):
        super().__init__(path, limit, is_valid)

    def __iter__(self):
        with open(self.path, "r") as f:
            data = json.load(f)

        for i in range(self.start, self.limit):
            # prompt = [data["prompts"][i], data["prompts"][i + 1]]
            yield data["prompts"][i]


# TODO 사실 reward model의 데이터로는 SFT model의 output을 사람이 직접 평가한 것을 써야 함
class RewardDataset(BaseDataset):
    def __init__(self, path, limit, is_valid=False):
        super().__init__(path, limit, is_valid)

    def __iter__(self):
        with open(self.path, "r") as f:
            data = json.load(f)

        for i in range(self.start, self.limit, 2):
            if i == self.limit - 1:
                continue
            prompt = [data["prompts"][i], data["prompts"][i + 1]]
            output = [data["outputs"][i], data["outputs"][i + 1]]
            scores = [data["scores"][i], data["scores"][i + 1]]
            yield [prompt, output, scores]


class SFTDataset(BaseDataset):
    def __init__(self, path, limit, is_valid=False):
        super().__init__(path, limit, is_valid)

    def __iter__(self):
        with open(self.path, "r") as f:
            data = json.load(f)

        for prompt, output in zip(data["prompts"][self.start : self.limit], data["outputs"][self.start : self.limit]):
            yield prompt + output


# Since Training lanugae models to follow instructions with human feedback used
# SFT : 13k
# RM : 33k
# RL : 31k
# respectively, we split datasets with ratio 1:2:2.
class AnthropicDataset:
    def __init__(self, conf):
        self.conf = conf
        if not os.path.exists(conf.dataset.save_path):
            os.makedirs(conf.dataset.save_path)
        self.sft_path = os.path.join(conf.dataset.save_path, conf.dataset.sft_path)
        self.rm_path = os.path.join(conf.dataset.save_path, conf.dataset.rm_path)
        self.rl_path = os.path.join(conf.dataset.save_path, conf.dataset.rl_path)
        self.do_sft = not os.path.exists(self.sft_path)
        self.do_rm = not os.path.exists(self.rm_path)
        self.do_rl = not os.path.exists(self.rl_path)
        if self.do_sft or self.do_rm or self.do_rl:
            self.dataset = load_dataset("Anthropic/hh-rlhf").shuffle()

    # TODO MAX SCORE 어떻게 할 지
    def _refine_dataset(self, chosen_data, rejected_data=None, label=None):
        if rejected_data:
            dataset = self._extract_with_rejected_data(chosen_data, rejected_data, label, score=1)
            return {
                "prompts": dataset[0],
                "outputs": dataset[1],
                "scores": dataset[2],
            }
        else:
            dataset = self._extract_data(chosen_data, label, score=1)
            # Shuffle
            dataset = np.array(dataset)
            random_idx = list(range(len(dataset[0])))
            random.shuffle(random_idx)
            return {
                "prompts": dataset[0][random_idx].tolist(),
                "outputs": dataset[1][random_idx].tolist(),
                "scores": dataset[2][random_idx].tolist(),
            }

    def _extract_with_rejected_data(self, chosen_data, rejected_dat, label=None, score=1):
        prefix = "\n\nAssistant: "
        chosen_chunk_list = [item.split("Assistant:") for item in chosen_data]
        rejected_chunk_list = [item.split("Assistant:") for item in rejected_dat]
        prompts, outputs, scores = [], [], []
        for chosen_chunks, rejected_chunks in tqdm(
            zip(chosen_chunk_list, rejected_chunk_list), total=len(chosen_chunk_list), desc=f"collecting {label} data"
        ):
            for i, chosen_chunk in enumerate(chosen_chunks[:-1]):
                if i == 0:
                    prompt = chosen_chunk
                else:
                    prompt += prefix + chosen_chunk.strip()
            chosen_output = prefix + chosen_chunks[-1]
            rejected_output = prefix + rejected_chunks[-1]
            prompts += [prompt, prompt]
            outputs += [chosen_output, rejected_output]
            scores += [score, 0]

        return [prompts, outputs, scores]

    def _extract_data(self, data, label=None, score=1):
        prefix = "\n\nAssistant: "
        chunk_list = [item.split("Assistant:") for item in data]
        prompts, outputs, scores = [], [], []
        for chunks in tqdm(chunk_list, total=len(chunk_list), desc=f"collecting {label} data"):
            for i, chunk in enumerate(chunks[:-1]):
                if i == 0:
                    prompt = chunk
                else:
                    prompt += prefix + chunk.strip()
            output = prefix + chunks[-1]
            prompts.append(prompt)
            outputs.append(output)
            scores.append(score)
        return [prompts, outputs, scores]

    def _save_json(self, path, selected_dataset, rejected_dataset=None, label=None):
        data_dict = self._refine_dataset(selected_dataset, rejected_dataset, label)
        with open(path, "w") as f:
            json.dump(data_dict, f)

    def save_dataset(self):
        # Check if it has to save dataset
        if not self.do_sft and not self.do_rm and not self.do_rl:
            return

        total_len = len(self.dataset["train"])
        len_sft = int(total_len * 0.2)
        len_rm = int(total_len * 0.4)
        len_rl = int(total_len * 0.4)

        # SFT
        if self.do_sft:
            dataset = self.dataset["train"][:len_sft]["chosen"]
            self._save_json(self.sft_path, dataset, label="SFT")
        length = len_sft

        # RM
        if self.do_rm:
            dataset = self.dataset["train"][length : length + len_rm]["chosen"]
            rejected_dataset = self.dataset["train"][length : length + len_rm]["rejected"]
            self._save_json(self.rm_path, dataset, rejected_dataset, label="RM")
        length = length + len_rm

        # RL
        if self.do_rl:
            dataset = self.dataset["train"][length : length + len_rl]["chosen"]
            self._save_json(self.rl_path, dataset, label="RL")
