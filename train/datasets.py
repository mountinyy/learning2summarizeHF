import os
import random

from torch.utils.data import Dataset

from datasets import load_dataset
from tqdm import tqdm
import json

class SFTDataset(Dataset):
    def __init__(self, dataset):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
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
        self.dataset = load_dataset("Anthropic/hh-rlhf").shuffle()
        
    # TODO MAX SCORE 어떻게 할 지
    def _refine_dataset(self, chosen_data, rejected_data=None, label=None):
        dataset = self._extract_data(chosen_data, label, score=5)
        if rejected_data:
            rejected_dataset = self._extract_data(rejected_data, label, score=0)
            dataset = [chosen + rejected for chosen, rejected in zip(dataset, rejected_dataset)]
        return {
            'prompts': dataset[0], 
            'outputs': dataset[1],
            'scores': dataset[2],
        }
    
    def _extract_data(self, data, label=None, score=5):
        prefix = "\n\nAssistant: "
        chunk_list = [item.split("Assistant:") for item in data]
        prompts, outputs, scores = [], [], []
        for chunks in tqdm(chunk_list, total=len(chunk_list), desc=f'collecting {label} data'):
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
        with open(path, 'w') as f:
            json.dump(data_dict, f)
            
    def save_dataset(self):
        total_len = len(self.dataset['train'])
        len_sft = int(total_len * 0.2)
        len_rm = int(total_len * 0.4)
        len_rl = int(total_len * 0.4)
        
        # SFt
        if not os.path.exists(self.sft_path):
            dataset = self.dataset['train'][:len_sft]['chosen']
            self._save_json(self.sft_path, dataset, label="SFT")
        length = len_sft
        
        # RM
        if not os.path.exists(self.rm_path):
            dataset = self.dataset['train'][length : length+len_rm]['chosen']
            rejected_dataset = self.dataset['train'][length : length+len_rm]['rejected']
            self._save_json(self.rm_path, dataset, rejected_dataset, label="RM")
        length = length + len_rm
        
        # RL
        if not os.path.exists(self.rl_path):
            dataset = self.dataset['train'][length : length+len_rl]['chosen']
            self._save_json(self.rl_path, dataset, label="RL")
        
        
        

