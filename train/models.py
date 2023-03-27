from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class Actor(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.model = AutoModelForCausalLM.from_pretrained(self.conf.sft.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf.sft.model_name, padding_side="left")

    def forward(self, states, states_mask):
        model_output = self.model.forward(states, attention_mask=states_mask, return_dict=True)
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
