"""DAgger SAPG/PPO agent — subclass of rl_games' A2CAgent that adds an MSE
imitation term to the standard PPO/SAPG update.

Two algo names registered (`dagger_sapg`, `dagger_ppo`) point at the same
class; the SAPG-vs-PPO distinction is purely yaml-configured (via
``coef_cond`` / ``intr_reward_coef`` / ``expl_blocks`` / ``off_policy_ratio``
/ ``central_value_config`` knobs already supported by A2CAgent).

Override surface is intentionally minimal:
    - ``__init__``      build the teacher + cache λ_D scheduler params
    - ``init_tensors``  allocate experience_buffer['teacher_actions'] and add to tensor_list
    - ``play_steps``    monkey-patch ``self.env_step`` to record teacher labels per step
    - ``prepare_dataset`` patch teacher_actions into the dataset values dict
    - ``calc_gradients`` copy of parent body with one line added to mix in λ_D · MSE

Loss is `(1 - λ_D) · L_PPO + λ_D · MSE(student_μ, teacher_μ)` with a linear
λ_D curriculum from `lambda_d_start → lambda_d_floor` over
`lambda_d_decay_frac × max_epochs`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common import common_losses

from .teacher import Teacher, latest_checkpoint


class DAggerA2CAgent(A2CAgent):
    """rl_games A2CAgent + DAgger MSE imitation term.

    Args (via ``params['config']['dagger']``):
        teacher_run_dir        Directory containing the teacher's rl_games checkpoint(s).
        teacher_checkpoint     Optional explicit .pth path; else newest under run_dir.
        teacher_task_id        Gym task id used to load the teacher (must match the
                               teacher's training env so num_actors / obs_shape align).
        teacher_agent_key      Hydra entry-point key for the teacher's train yaml.
        lambda_d_start         λ_D at epoch 0.
        lambda_d_floor         λ_D after the decay window.
        lambda_d_decay_frac    Fraction of max_epochs over which λ_D linearly decays.
    """

    def __init__(self, base_name: str, params: dict):
        super().__init__(base_name, params)

        cfg = params["config"]["dagger"]
        ckpt = Path(cfg["teacher_checkpoint"]) if cfg.get("teacher_checkpoint") else latest_checkpoint(cfg["teacher_run_dir"])

        # Teacher was trained on the env's *actor obs* (obs_list, ~140-d). The
        # live env exposes that as ``teacher_obs`` in addition to the student's
        # depth+proprio policy obs. The central ``RlGamesVecEnvWrapper`` knows
        # both — read env_info from it via ``teacher_env_info`` (single source
        # of truth for shapes AND yaml-derived clip bounds).
        from isaacsimenvs.utils.rlgames_utils import teacher_env_info
        wrapped = self.vec_env.env if hasattr(self.vec_env, "env") else self.vec_env
        teacher_info = teacher_env_info(wrapped) if hasattr(wrapped, "teacher_obs_space") else None

        self.teacher = Teacher(
            task_id=cfg.get("teacher_task_id", params["config"]["env_name"]),
            agent_key=cfg.get("teacher_agent_key", "rl_games_sapg_cfg_entry_point"),
            checkpoint_path=ckpt,
            num_envs=self.num_actors,
            rl_device=str(self.ppo_device),
            env_info=teacher_info,
        )
        # SAPG default is `linspace(50, 0, num_envs)`, but the network's block
        # lookup is exact-equality argmax against {50, 40, 30, 20, 10, 0} so
        # off-grid values fall through to block 0 (=50) anyway. Pin explicitly;
        # tune via `teacher_block_id` in the dagger yaml.
        self.teacher.pin_block_id(float(cfg.get("teacher_block_id", 50.0)))

        # Teacher reads `self.obs["teacher"]` (the un-augmented actor obs that
        # `DAggerRlGamesVecEnvWrapper` passes through). SAPG's block-id is
        # appended internally by `Teacher.get_action`.
        self._teacher_obs_key = "teacher"

        self.lambda_d_start = float(cfg.get("lambda_d_start", 1.0))
        self.lambda_d_floor = float(cfg.get("lambda_d_floor", 0.1))
        self.lambda_d_decay_frac = float(cfg.get("lambda_d_decay_frac", 0.5))

        # If True, training rollouts use the student's deterministic μ instead
        # of sampling from N(μ, σ). Still DAgger (student visits states, teacher
        # labels), but with no exploration noise. Useful when σ never gets
        # gradient (pure-BC), avoids ±0.37 jitter on every action. PPO ratio
        # stays = 1 (replay log_prob computed at μ matches stored log_prob at μ).
        self.deterministic_rollouts = bool(cfg.get("deterministic_rollouts", False))

        # Counter incremented inside the wrapped env_step (one per env-step).
        self._dagger_step_counter = 0
        self._dagger_buffer_allocated = False

    # ----- buffers -----

    def init_tensors(self):
        super().init_tensors()
        # Allocate teacher_actions tensor inside the experience buffer (same
        # (T, B, A) shape as 'actions'), so prepare_dataset / dataset slicing
        # treats it like any other rollout tensor.
        if not self._dagger_buffer_allocated:
            self.experience_buffer.tensor_dict["teacher_actions"] = torch.zeros(
                (self.horizon_length, self.num_actors, self.actions_num),
                dtype=torch.float32,
                device=self.ppo_device,
            )
            # Make sure swap_and_flatten01 sees this key when building batch_dict.
            if "teacher_actions" not in self.tensor_list:
                self.tensor_list = list(self.tensor_list) + ["teacher_actions"]
            self._dagger_buffer_allocated = True

    # ----- deterministic rollout override -----

    def get_action_values(self, obs, *args, **kwargs):
        """Override to use student's μ as the rollout action when configured."""
        res_dict = super().get_action_values(obs, *args, **kwargs)
        if self.deterministic_rollouts:
            res_dict["actions"] = res_dict["mus"]
        return res_dict

    # ----- λ_D schedule -----

    def _lambda_d(self) -> float:
        end_epoch = self.lambda_d_decay_frac * float(self.max_epochs)
        if end_epoch <= 0 or self.epoch_num >= end_epoch:
            return self.lambda_d_floor
        frac = float(self.epoch_num) / end_epoch
        return self.lambda_d_start + (self.lambda_d_floor - self.lambda_d_start) * frac

    # ----- rollout: monkey-patch env_step to record teacher labels -----

    def play_steps(self):
        # Make sure teacher_actions buffer exists (defensive — should be set by init_tensors).
        if "teacher_actions" not in self.experience_buffer.tensor_dict:
            self.init_tensors()

        self._dagger_step_counter = 0
        original_env_step = self.env_step

        def wrapped_env_step(actions):
            # Query the teacher on the SAME obs we just fed to get_action_values,
            # so its RNN advances in lock-step with the student's.
            base_obs = self.obs[self._teacher_obs_key]
            with torch.no_grad():
                teacher_a = self.teacher.get_action(base_obs)
            n = self._dagger_step_counter
            self.experience_buffer.tensor_dict["teacher_actions"][n].copy_(teacher_a)
            self._dagger_step_counter += 1

            result = original_env_step(actions)
            # Reset teacher RNN on env dones — same indices the parent uses for its own RNN reset.
            dones = result[3]
            done_idx = dones.nonzero(as_tuple=False).flatten()
            if done_idx.numel() > 0:
                self.teacher.reset_idx(done_idx)
            return result

        self.env_step = wrapped_env_step
        try:
            return super().play_steps()
        finally:
            self.env_step = original_env_step

    # ----- dataset: ensure teacher_actions reaches calc_gradients via input_dict -----

    def prepare_dataset(self, batch_dict, train_value_mean_std=True):
        super().prepare_dataset(batch_dict, train_value_mean_std=train_value_mean_std)
        # Patch teacher_actions into the same flat ordering as the rest of the dataset.
        # batch_dict was built with swap_and_flatten01 → shape (T*B, A) for our key.
        self.dataset.values_dict["teacher_actions"] = batch_dict["teacher_actions"]

    # ----- per-minibatch loss: copy of A2CAgent.calc_gradients with the DAgger term added -----

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)
        teacher_actions_batch = input_dict["teacher_actions"]   # (mb, action_dim)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length
            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros((len(values), 1), device=self.ppo_device)
            if self.bound_loss_type == "regularisation":
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == "bound":
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(len(mu), device=self.ppo_device)

            if self.expl_type.startswith("mixed_expl") and self.config.get("expl_reward_type") == "entropy":
                ec_candidates = self.intr_reward_coef[:: self.intr_coef_block_size]
                ec_identifiers = self.intr_reward_coef_embd[:: self.intr_coef_block_size, 0].reshape(-1, 1)
                ec_indices = torch.argmax((obs_batch[:, -self.intr_reward_coef_embd.shape[1]] == ec_identifiers).float(), dim=0)
                entropy_coef = ec_candidates[ec_indices]
            elif self.expl_type.startswith("simple") and self.config.get("expl_reward_type") == "entropy":
                entropy_coef = self.intr_reward_coef
            else:
                entropy_coef = self.entropy_coef

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, (entropy_coef * entropy).unsqueeze(1), b_loss.unsqueeze(1)],
                rnn_masks,
            )
            a_loss, c_loss, entropy_loss, b_loss = losses[0], losses[1], losses[2], losses[3]

            ppo_loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy_loss + b_loss * self.bounds_loss_coef

            # ---- DAgger MSE term ----
            # Per-element MSE (no reduction yet) so we can mask + average like the other losses.
            mse_per_elem = (mu - teacher_actions_batch).pow(2).mean(dim=-1, keepdim=True)  # (mb, 1)
            (mse_loss,), _ = torch_ext.apply_masks([mse_per_elem], rnn_masks)

            lam_d = self._lambda_d()
            loss = (1.0 - lam_d) * ppo_loss + lam_d * mse_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        all_grads = self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(
            self,
            {
                "values": value_preds_batch,
                "returns": return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks": rnn_masks,
            },
            curr_e_clip,
            0,
        )

        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        contrib = torch.logical_and(ratio < 1.0 + curr_e_clip, ratio > 1.0 - curr_e_clip).float()

        extras = {
            "on_policy_contrib": contrib.mean().item(),
            "off_policy_contrib": 0,
            "on_policy_grads": all_grads.detach().cpu(),
            "off_policy_grads": torch.zeros_like(all_grads).cpu(),
        }
        # rl_games' train_epoch only forwards specific keys from `extras` to
        # the writer (a2c_common.py:1421-1426). Custom keys are dropped, so
        # write DAgger scalars directly to the SummaryWriter (synced to wandb
        # via tb-mirroring). `self.frame` is the global env-frame counter.
        if self.writer is not None:
            self.writer.add_scalar("dagger/L_D", mse_loss.detach().item(), self.frame)
            self.writer.add_scalar("dagger/lambda_d", lam_d, self.frame)
            self.writer.add_scalar("dagger/teacher_action_l2",
                                   teacher_actions_batch.detach().pow(2).mean().sqrt().item(),
                                   self.frame)
            self.writer.add_scalar("dagger/student_action_l2",
                                   mu.detach().pow(2).mean().sqrt().item(),
                                   self.frame)
        if self.expl_type.startswith("mixed_expl") and self.intr_reward_coef_embd is not None:
            bl_ids = self.intr_reward_coef_embd[:: self.intr_coef_block_size, 0].reshape(-1, 1)
            bl_idxs = torch.argmax((obs_batch[:, -self.intr_reward_coef_embd.shape[1]] == bl_ids).float(), dim=0)
            extras["entropies"] = [
                torch.nan_to_num(entropy[bl_idxs == i].detach().mean()).item()
                for i in range(self.num_actors // self.intr_coef_block_size)
            ]

        self.train_result = (
            a_loss,
            c_loss,
            torch_ext.apply_masks([entropy.unsqueeze(1)], rnn_masks)[0][0],
            kl_dist,
            self.last_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
            extras,
        )
