from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Experience:
    reward: torch.Tensor
    old_action_prob: torch.Tensor
    old_value: torch.Tensor
    sequences: torch.Tensor
    advantage: torch.Tensor

    def to_device(self, device: torch.device):
        self.reward = self.reward.to(device)
        self.old_action_prob = self.old_action_prob.to(device)
        self.old_value = self.old_value.to(device)
        self.sequences = self.sequences.to(device)
        self.advantage = self.advantage.to(device)


class ExperienceController:
    def __init__(self, sft, rm, rl, critic, kl_coef, pad_token_id, eos_token_id):
        self.sft = sft
        self.rm = rm
        self.rl = rl
        self.critic = critic
        self.kl_coef = kl_coef
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def create_experience(self, states, states_mask):
        self.sft.eval()
        self.rm.eval()
        self.rl.eval()
        self.critic.eval()

        # TODO attention mask를 추가해야 할까?
        # https://github.com/hpcaitech/ColossalAI/blob/main/applications/Chat/coati/experience_maker/naive.py
        # https://github.com/hpcaitech/ColossalAI/blob/1c7734bc94ac1a7215e08368adc4e7e25e3b8102/applications/Chat/coati/models/base/actor.py#L35-L38
        # 여기서 generate()가 attention mask들도 리턴하는 것 고려.
        actions, total_actions = self.rl.generate(states, states_mask)
        attention_mask = actions.not_equal(self.pad_token_id).to(dtype=torch.long, device=actions.device)
        total_action_probs = self.rl(total_actions, attention_mask)
        total_action_probs = self.log_prob(total_action_probs, total_action_probs)
        base_action_probs = self.sft(total_actions, attention_mask)
        base_action_probs = self.log_porb(base_action_probs, base_action_probs)
        value = self.critic(actions)
        r = self.rm(actions)
        kl = self.compute_kl(total_action_probs, base_action_probs)
        reward = r - self.kl_coef * kl
        advantage = reward - value

        return Experience(reward, total_action_probs, value, actions, advantage)

        # Colossal AI에서는 (bz, seq, vocab) probs에서 실제 정답인 vocab들만 추려서 비교함
        # 근데 어차피 SFT와의 분포를 비교하는 거면 전체 vocab에 대해서 계산해도 되지 않을까?

    def log_prob(self, sequences, labels):
        log_probs = F.log_softmax(sequences, dim=-1)
        log_probs = log_probs[:, :-1, :].gather(dim=-1, index=labels[:, 1:]).squeeze(-1)
        return log_probs

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
