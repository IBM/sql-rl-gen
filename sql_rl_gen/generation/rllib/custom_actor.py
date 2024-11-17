from textrl.actor import TextRLActor, TextPPO
from pfrl.utils.batch_states import batch_states
from sql_rl_gen.generation.envs.utils import find_device

device_to_use = find_device()

class CustomActor(TextRLActor):
    def __init__(self, env, model, tokenizer, optimizer='sgd', gpu_id=0, unfreeze_layer_from_past=0, act_deterministically=True, temperature=1.0, top_k=0, top_p=1.0):
        super().__init__(env, model, tokenizer, optimizer, gpu_id, unfreeze_layer_from_past, act_deterministically, temperature, top_k, top_p)
        self.device = device_to_use

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