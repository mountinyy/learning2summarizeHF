from dataclasses import dataclass

import torch

from utils.computation import convert_by_tokenizer, log_prob


@dataclass
class Experience:
    reward: torch.Tensor
    old_action_prob: torch.Tensor
    action_attention_mask: torch.Tensor
    old_value: torch.Tensor
    actions: torch.Tensor
    advantage: torch.Tensor
    critic_actions: torch.Tensor

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.reward = self.reward.to(device)
        self.old_action_prob = self.old_action_prob.to(device)
        self.action_attention_mask = self.action_attention_mask.to(device)
        self.old_value = self.old_value.to(device)
        self.actions = self.actions.to(device)
        self.advantage = self.advantage.to(device)
        self.critic_actions = self.critic_actions.to(device)

        return self


class ExperienceController:
    def __init__(self, sft, rm, rl, critic, kl_coef):
        self.sft = sft
        self.rm = rm
        self.rl = rl
        self.critic = critic
        self.kl_coef = kl_coef
        self.pad_token_id = self.rl.tokenizer.pad_token_id
        self.eos_token_id = self.rl.tokenizer.eos_token_id

    @torch.no_grad()
    def create_experience(self, data, device):
        """experience를 생성한다.
        주어진 data를 state로 우리가 훈련하는 RL이 먼저 action을 취한다(generate())
        이를 기반으로 RL과 base_model인 SFT가 각각 transition probability를 구한다(forward())

        이후 critic은 action만을 보고 평가하고, reward_model은 전체 action을 보고 평가한다.
        critic, action의 reward와 RL, SFT의 probability를 기반으로 REWARD를 계산하고, value를 빼줌으로써
        Advantage를 구한다.

        Args:
            data (_type_): _description_
            device (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.sft.eval()
        self.rm.eval()
        self.rl.eval()
        self.critic.eval()

        tokenized_data = self.rl.tokenizer(data, padding=True, truncation=True, return_tensors="pt").to(device)
        states = tokenized_data["input_ids"]
        states_mask = tokenized_data["attention_mask"]

        # TODO attention mask를 추가해야 할까?
        # https://github.com/hpcaitech/ColossalAI/blob/main/applications/Chat/coati/experience_maker/naive.py
        # https://github.com/hpcaitech/ColossalAI/blob/1c7734bc94ac1a7215e08368adc4e7e25e3b8102/applications/Chat/coati/models/base/actor.py#L35-L38
        # 여기서 generate()가 attention mask들도 리턴하는 것 고려.
        actions, total_actions = self.rl.generate(states, states_mask, pad_token_id=self.eos_token_id)
        attention_mask = total_actions.not_equal(self.pad_token_id).to(dtype=torch.long, device=actions.device)
        total_action_probs = self.rl(total_actions, attention_mask)
        total_action_probs = log_prob(total_action_probs, total_actions)
        base_action_probs = self.sft(total_actions, attention_mask)
        base_action_probs = log_prob(base_action_probs, total_actions)

        critic_actions = convert_by_tokenizer(actions, self.rl.tokenizer, self.critic.tokenizer, device)
        critic_total_actions = convert_by_tokenizer(total_actions, self.rl.tokenizer, self.critic.tokenizer, device)
        # action_mask = self.compute_action_mask(total_actions, actions.size(-1))
        # value = self.critic(critic_actions, action_mask)
        # TODO critic으로 value 계산할 때 action만 고려해서 계산하는지 total_action을 모두 고려해서 계산하는지
        value = self.critic(critic_actions)
        # r = self.rm(critic_actions)
        r = self.rm(critic_total_actions)
        kl = self.compute_kl(total_action_probs, base_action_probs)
        reward = r - self.kl_coef * kl
        advantage = reward - value
        return Experience(reward, total_action_probs, attention_mask, value, total_actions, advantage, critic_actions)

        # Colossal AI에서는 (bz, seq, vocab) probs에서 실제 정답인 vocab들만 추려서 비교함
        # 근데 어차피 SFT와의 분포를 비교하는 거면 전체 vocab에 대해서 계산해도 되지 않을까?

    def convert_by_tokenizer(self, sequences):
        sentence = self.rl.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        tokenized = self.critic.tokenizer(sentence, padding=True, return_tensors="pt")["input_ids"]
        return tokenized.to(sequences.device)

    def compute_action_mask(self, sequences, state_len):
        action_mask = torch.ones_like(sequences, dtype=torch.bool)
        for i in range(action_mask.size(0)):
            action_mask[:, :state_len] = False
        # action_mask = action_mask[:, 1:]
        return action_mask

    # TODO KL divergence의 approximation
    # 왜 이게 좋은지 알고 싶다면 http://joschu.net/blog/kl-approx.html 참조
    def compute_kl(self, total_action_probs, base_action_probs):
        ratio = total_action_probs - base_action_probs
        kl = (ratio.exp() - 1) - ratio
        # TODO 만약 action mask가 있다면 그에 대한 처리도 해줘야 함 (아래 참조)
        # https://github.com/hpcaitech/ColossalAI/blob/1c7734bc94ac1a7215e08368adc4e7e25e3b8102/applications/Chat/coati/models/utils.py#L24-L26
        # TODO dim=1이 맞나? 확인은 안해봄.
        kl = kl.mean(dim=1)
        return kl
