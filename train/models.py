from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class Actor(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.model = AutoModelForCausalLM.from_pretrained(self.conf.sft.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.conf.sft.model_name, padding_side="left", max_model_lengt=conf.sft.max_seq_length
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def forward(self, states, states_mask):
        model_output = self.model(states, attention_mask=states_mask, return_dict=True)
        total_action_probs = model_output.logits
        return total_action_probs

    def generate(self, states, states_mask):
        allowed_gen_length = self.conf.sft.max_seq_length - states.shape[1]
        if allowed_gen_length < self.conf.min_gen_length:
            raise ValueError(
                f"Prompt with length {states.shape[1]} is too long!(Maximum length : {self.conf.max_seq_length})"
            )

        total_actions = self.model.generate(
            input_ids=states,
            attention_mask=states_mask,
            temperature=self.conf.sft.temperature,
            max_new_tokens=self.conf.sft.max_gen_length,
            no_repeat_ngram_size=3,
        )
        actions = total_actions[:, states.shape[1] :]

        return (actions, total_actions)


class Critic(nn.Module):
    def __init__(self, conf, is_reward):
        super().__init__()
        self.conf = conf

        self.model = AutoModel.from_pretrained(self.conf.reward.model_name)
        model_hidden_dim = self.model.config.hidden_size
        self.head_lm = nn.Sequential(
            nn.Linear(model_hidden_dim, self.conf.reward.hidden_dim),
            nn.GELU(),
            nn.Linear(self.conf.reward.hidden_dim, 1),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.conf.reward.model_name,
            padding_side="left",
            padding=True,
            truncation=True,
            model_max_length=self.conf.common.max_seq_length,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        reward = self.head_lm(output.last_hidden_state)

        return reward
