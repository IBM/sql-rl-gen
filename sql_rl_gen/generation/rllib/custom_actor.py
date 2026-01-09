import pfrl
import torch
from textrl.actor import TextRLActor, TextPPO, SoftmaxCategoricalHead
from pfrl.utils.batch_states import batch_states
from sql_rl_gen.generation.envs.utils import find_device

device_to_use = find_device()

class CustomActor(TextRLActor):
    def __init__(self, env, model, tokenizer, optimizer='sgd', gpu_id=0, unfreeze_layer_from_past=0, act_deterministically=True, temperature=1.0, top_k=0, top_p=1.0):
        super().__init__(env, model, tokenizer, optimizer, gpu_id, unfreeze_layer_from_past, act_deterministically, temperature, top_k, top_p)
        self.device = device_to_use

    def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20, lr=3e-6):
        """Device-aware PPO builder.

        Upstream textrl's `TextRLActor.agent_ppo()` hard-calls `.cuda()`, which breaks
        CPU-only/offline environments. This override keeps the same architecture but
        respects the selected device.
        """

        policy = torch.nn.Sequential(
            self.middle_model,
            self.remaining_model,
            self.converter,
            SoftmaxCategoricalHead(
                self.env,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            ),
        )
        vf = torch.nn.Sequential(
            torch.nn.Linear(self.obs_size, self.obs_size // 2),
            torch.nn.Linear(self.obs_size // 2, self.obs_size // 4),
            torch.nn.Linear(self.obs_size // 4, 1),
        )
        model = pfrl.nn.Branched(policy, vf)

        if isinstance(self.optimizer, str):
            if self.optimizer.lower() == "adamw":
                opt = torch.optim.AdamW(model.parameters(), lr=lr)
            else:
                opt = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            opt = self.optimizer

        device = self.device
        if device.type not in ("cpu", "cuda"):
            device = torch.device("cpu")
        model = model.to(device)

        agent = TextPPO(
            model,
            opt,
            gpu=(self.gpu_id if device.type == "cuda" else None),
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            clip_eps_vf=None,
            entropy_coef=0,
            gamma=0.95,  # https://arxiv.org/abs/2210.01241
            lambd=1,
            max_grad_norm=1.0,
            standardize_advantages=True,
            act_deterministically=self.act_deterministically,
        )
        self.agent = agent
        return agent

class CustomTextPPO(TextPPO):
    def __init__(self, model, optimizer, device, obs_normalizer=None, gamma=0.99, lambd=0.95, phi=lambda x: x, value_func_coef=1.0,
                 entropy_coef=0.01, update_interval=2048, minibatch_size=64, epochs=10, clip_eps=0.2, clip_eps_vf=None, standardize_advantages=True,
                 batch_states=batch_states, recurrent=False, max_recurrent_sequence_len=None, act_deterministically=False, max_grad_norm=None,
                 value_stats_window=1000, entropy_stats_window=1000, value_loss_stats_window=100, policy_loss_stats_window=100):
        super().__init__(model, optimizer, obs_normalizer, None, gamma, lambd, phi, value_func_coef, entropy_coef,
                         update_interval, minibatch_size, epochs, clip_eps, clip_eps_vf, standardize_advantages,
                         batch_states, recurrent, max_recurrent_sequence_len, act_deterministically, max_grad_norm,
                         value_stats_window, entropy_stats_window, value_loss_stats_window, policy_loss_stats_window)
        self.device = device
        self.model.to(self.device)
        if self.obs_normalizer is not None:
            self.obs_normalizer.to(self.device)
