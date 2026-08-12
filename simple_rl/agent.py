import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch import nn, optim

from simple_rl.player import InferenceConfig
from simple_rl.utils import (
    asymmetric_critic,
    datasets,
    models,
    schedulers,
    torch_utils,
)
from simple_rl.utils.asymmetric_critic import AsymmetricCriticConfig
from simple_rl.utils.conditioning_utils import CONDITIONING_IDX_DIM
from simple_rl.utils.experience import ExperienceBuffer
from simple_rl.utils.network import NetworkConfig
from simple_rl.utils.observer import DefaultAlgoObserver
from simple_rl.utils.rewards_shaper import (
    DefaultRewardsShaper,
    RewardsShaperParams,
)
from simple_rl.utils.sapg_utils import filter_leader, shuffle_batch


@dataclass
class SapgConfig:
    """Configuration for SAPG (Split and Aggregate Policy Gradients).

    M blocks of environments each receive a different entropy coefficient (linspace 0.5→0.0).
    A learned (M, conditioning_dim) embedding matrix distinguishes blocks; the integer
    conditioning index is appended to obs and looked up at forward time.

    Obs normalization note: simple_rl normalizes only the raw obs dims (excluding the integer
    conditioning index), so the per-block conditioning lookup is always exact.  rl_games uses
    the same design via coef_id_idx: only obs[:, :coef_id_idx] is normalized and the appended
    float block identifier (50.0, 40.0, ...) passes through unnormalized.  Both frameworks
    correctly exclude the block identifier from normalization.

    Set ppo.sapg to enable. For EPO, also set ppo.epo.
    """

    num_conditionings: int
    """M in the SAPG paper — number of blocks = number of distinct entropy levels.
    Envs are split into M equal groups; each group receives its own conditioning embedding
    and entropy coefficient.  Must evenly divide num_actors."""

    conditioning_dim: int
    """Dimension of the per-block learnable embedding appended to obs.
    Corresponds to rl_games' expl_reward_coef_embd_size (default 32).
    The policy network input grows by this many dims at inference time."""

    use_others_experience: bool = True
    """If True, each minibatch is augmented with re-labelled experience from the leader block
    (rl_games 'lf' / leader-follower mode).  Disabled for very small env counts where
    the augmented batch size would not divide evenly."""

    recurrent_experience_sharing: Literal["reject", "reuse_state"] = "reject"
    """How recurrent memory is handled when leader/follower sharing is enabled.

    ``reject`` is the safe default because a saved LSTM/transformer state was
    produced under the trajectory's original conditioning. ``reuse_state``
    explicitly opts into the approximation used by the reference SAPG code:
    observations are relabelled, while the saved recurrent state is reused.
    This is empirically useful but is not an exactly on-policy history.
    """

    off_policy_ratio: int = 1
    """Number of extra off-policy blocks added per minibatch when use_others_experience=True.
    Each extra block increases the effective minibatch by batch_size / M.
    Value of 1 means one additional off-policy copy (2× block-0 data per update)."""

    # Block ordering (same convention as rl_games):
    #   block 0     = LEADER   — highest entropy coef = 0.5 * entropy_coef_scale  (most exploratory)
    #   block M-1   = FOLLOWER — entropy coef = 0.0                               (pure exploitation)
    #   blocks 1…M-2           — linearly interpolated between leader and follower
    #
    # Entropy coefficients: torch.linspace(0.5, 0.0, M) * entropy_coef_scale
    # E.g. M=6, scale=0.005 → coefs [0.0025, 0.002, 0.0015, 0.001, 0.0005, 0.0]
    use_entropy_bonus: bool = True
    """Whether to add a per-block entropy bonus to each block's policy loss.
    True (default): block k gets entropy coef = linspace(0.5, 0.0, M)[k] * entropy_coef_scale.
    False: all blocks are trained with no entropy bonus (equivalent to SAPG without exploration)."""

    entropy_coef_scale: float = 0.005
    """Scale multiplier applied to the linspace(0.5, 0.0, M) entropy coefficient schedule.
    Actual per-block coefs = linspace(0.5, 0.0, M) * entropy_coef_scale.
    Block 0 (leader) gets the maximum coef = 0.5 * entropy_coef_scale.
    Block M-1 (follower) always gets 0.0 regardless of scale.
    Pretrained policy uses 0.005 → leader coef = 0.0025."""

    entropy_in_returns: bool = False
    """If True, adds entropy * coef to the shaped rewards before GAE computation (intrinsic reward).
    If False (default, matches rl_games), entropy only enters via the loss term directly.
    Setting True makes the entropy bonus propagate through value targets, which can help
    exploration on sparse-reward tasks but may slow convergence on dense-reward ones."""


@dataclass
class EpoConfig:
    """Configuration for EPO (Evolutionary Policy Optimization).

    EPO extends SAPG by periodically ranking blocks by episode reward, killing the
    worst-performing fraction, and replacing them with pairwise-averaged conditioning
    embeddings of the survivors. Requires ppo.sapg to also be set (EPO is SAPG + evolution).
    """

    evolution_frequency: int = 50
    """Run the evolutionary update every N training epochs.
    Reference EPO uses frame-based intervals of 20M frames with a 200M-frame warm-up;
    at 24576 envs × 16 horizon that is ≈51 epochs per interval.
    Our default of 50 epochs is calibrated for large-scale runs (24576 envs); it fires
    more frequently at smaller env counts (e.g. every ~5M frames at 6144 envs)."""

    evolution_kill_ratio: float = 0.02
    """Fraction of blocks to kill and replace per evolutionary update.
    Killed blocks have their conditioning embeddings replaced with pairwise-averaged
    embeddings of the surviving (top-ranked) blocks.
    NOTE: the reference EPO implementation (EPOObserver in EPO/isaacgymenvs/utils/rlgames_utils.py)
    hardcodes killing exactly 1 block per update regardless of M (effective ratio ≈ 1/M ≈ 0.016
    for M=64). The conservative 0.02 default replaces one block when M=64."""

    log_sigma: bool = False
    """Log mean action standard deviation (sigma) to TensorBoard each epoch.
    Useful for diagnosing whether per-block sigmas are collapsing or diverging."""

    log_off_policy_grads: bool = False
    """Log cosine similarity between on-policy and off-policy gradient directions each epoch.
    Expensive (requires two backward passes); only enable for debugging."""


@dataclass
class PpoConfig:
    """PPO training configuration. Required fields have no defaults; optional fields do."""

    # ── Required fields ───────────────────────────────────────────────────────
    num_actors: int
    learning_rate: float
    entropy_coef: float
    """Coefficient on the entropy bonus in the policy loss (encourages exploration).
    0.0 = no bonus; positive values push the policy toward higher-entropy actions."""

    horizon_length: int
    """Number of environment steps collected per environment before each gradient update.
    Total batch size = num_actors × horizon_length."""

    normalize_advantage: bool
    """Normalize advantages to mean=0, std=1 within each minibatch before the PPO update.
    Almost always beneficial for stability."""

    normalize_input: bool
    """Maintain a running mean/std of observations and normalize them online.
    Greatly helps when observation scales vary widely across dims."""

    grad_norm: float
    """Gradient norm clipping threshold. Gradients are rescaled to have L2 norm ≤ grad_norm."""

    critic_coef: float
    """Weight of the value (critic) loss in the combined loss: total = actor + critic_coef * critic."""

    gamma: float
    """Discount factor for future rewards. Typical value: 0.99."""

    tau: float
    """GAE-λ: controls the bias–variance trade-off in advantage estimation.
    τ=1.0 → Monte-Carlo returns (low bias, high variance).
    τ=0.0 → TD(0) (high bias, low variance).
    Typical value: 0.95."""

    reward_shaper: RewardsShaperParams
    mini_epochs: int
    """Number of gradient update passes over the collected rollout per epoch.
    Higher values extract more signal per sample but can cause policy divergence for LSTM."""

    e_clip: float
    """PPO clipping parameter ε. The policy ratio r = π_new/π_old is clipped to [1-ε, 1+ε].
    Smaller values → more conservative updates. Typical: 0.1–0.2."""

    # ── Fields with defaults ──────────────────────────────────────────────────
    multi_gpu: bool = False
    device: str = "cuda:0"
    weight_decay: float = 0.0
    asymmetric_critic: Optional[AsymmetricCriticConfig] = None
    truncate_grads: bool = False
    """If True, clip the total gradient norm to ``grad_norm`` before stepping."""

    save_frequency: int = 0
    save_best_after: int = 100
    print_stats: bool = True
    max_epochs: int = -1
    max_frames: int = -1
    max_wall_time_seconds: float = -1.0
    """Stop cleanly at the first epoch boundary after this many wall-clock seconds.
    A negative value disables the limit. The final checkpoint is always written."""
    lr_schedule: Optional[str] = None
    """Learning rate schedule type. None = constant. "adaptive" = adjust based on KL divergence
    (requires kl_threshold). "linear" = linear decay to 0 over max_epochs/max_frames."""

    schedule_type: Literal["legacy", "standard"] = "legacy"
    """Controls when the LR schedule steps are applied.
    "legacy": LR updates happen inside each minibatch loop (per mini-epoch step).
    "standard": LR updates happen once per full epoch after all mini-epochs."""

    kl_threshold: Optional[float] = None
    """Target KL divergence for adaptive LR schedule. If mean KL > threshold, LR is decreased;
    if KL < threshold/2, LR is increased. Only used when lr_schedule="adaptive"."""

    seq_length: int = 4
    """LSTM sequence length: the rollout is split into chunks of this length for BPTT.
    Should equal horizon_length for full-horizon backprop (the typical setting for LSTM policies).
    Shorter values trade gradient quality for memory."""

    zero_rnn_on_done: bool = True
    """Reset LSTM hidden state to zeros at episode boundaries (done=True).
    Almost always True; setting False lets the LSTM carry state across episode resets."""

    normalize_value: bool = False
    """Normalize value targets with a running mean/std before the critic update.
    Helps when reward magnitudes change over training."""

    games_to_track: int = 100
    """Rolling window size for the mean episode reward reported during training."""

    minibatch_size_per_env: int = 0
    minibatch_size: Optional[int] = None
    """Size of each gradient step. batch_size / minibatch_size minibatches are processed per
    mini-epoch. Smaller → more gradient steps per epoch but higher variance per step.
    Reference SAPG/EPO repos use 4 × num_envs (4 minibatches per epoch)."""

    mixed_precision: bool = False
    bounds_loss_coef: Optional[float] = None
    """Coefficient for an action-bounds regularization loss that penalizes actions outside
    [-1, 1] (before tanh squashing). None or 0.0 = disabled."""

    bound_loss_type: Literal["bound", "regularisation"] = "bound"
    value_bootstrap: bool = False
    """If True, bootstrap the value at episode truncations (timeout) rather than treating
    them as terminal states. Corrects value targets when episodes end due to time limits."""

    clip_actions: bool = True
    schedule_entropy: bool = False
    freeze_critic: bool = False

    # ── SAPG / EPO — both None means plain PPO ────────────────────────────────
    sapg: Optional[SapgConfig] = None
    """Set to a SapgConfig to enable SAPG (multi-block training with entropy diversity).
    Also required when using EPO."""

    epo: Optional[EpoConfig] = None
    """Set to an EpoConfig to add EPO evolutionary updates on top of SAPG.
    Requires sapg to also be set."""

    def to_inference_config(self) -> InferenceConfig:
        return InferenceConfig(
            normalize_input=self.normalize_input,
            clip_actions=self.clip_actions,
            device=self.device,
            normalize_value=self.normalize_value,
            conditioning_dim=self.sapg.conditioning_dim
            if self.sapg is not None
            else None,
            num_conditionings=self.sapg.num_conditionings
            if self.sapg is not None
            else None,
        )


def swap_and_flatten01(arr: torch.Tensor) -> torch.Tensor:
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


def print_statistics(
    print_stats: bool,
    curr_frames: float,
    step_time: float,
    step_inference_time: float,
    total_time: float,
    epoch_num: int,
    max_epochs: int,
    frame: float,
    max_frames: int,
) -> None:
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        if max_epochs == -1 and max_frames == -1:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}"
            )
        elif max_epochs == -1:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{max_frames:.0f}"
            )
        elif max_frames == -1:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}"
            )
        else:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}/{max_frames:.0f}"
            )


class Agent:
    def _validate_config(self) -> None:
        cfg = self.cfg
        positive_ints = {
            "num_actors": cfg.num_actors,
            "horizon_length": cfg.horizon_length,
            "seq_length": cfg.seq_length,
            "mini_epochs": cfg.mini_epochs,
            "games_to_track": cfg.games_to_track,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        if cfg.learning_rate <= 0 or cfg.grad_norm <= 0 or cfg.e_clip <= 0:
            raise ValueError("learning_rate, grad_norm, and e_clip must be positive")
        if not 0 <= cfg.gamma <= 1 or not 0 <= cfg.tau <= 1:
            raise ValueError("gamma and tau must be in [0, 1]")
        if cfg.max_wall_time_seconds == 0:
            raise ValueError("max_wall_time_seconds must be negative or positive")
        if hasattr(self.env, "num_envs") and self.env.num_envs != cfg.num_actors:
            raise ValueError(
                f"env.num_envs ({self.env.num_envs}) must equal num_actors "
                f"({cfg.num_actors})"
            )

        if cfg.epo is not None and cfg.sapg is None:
            raise ValueError("epo requires a sapg configuration")
        if cfg.sapg is None:
            return

        sapg = cfg.sapg
        if self.num_agents != 1:
            raise ValueError(
                "SAPG/EPO currently requires exactly one agent per environment"
            )
        if sapg.num_conditionings <= 0 or sapg.conditioning_dim <= 0:
            raise ValueError("SAPG conditioning counts and dimensions must be positive")
        if cfg.num_actors < sapg.num_conditionings:
            raise ValueError("num_actors must be at least num_conditionings")
        if cfg.num_actors % sapg.num_conditionings != 0:
            raise ValueError("num_conditionings must evenly divide num_actors")
        if not isinstance(sapg.off_policy_ratio, int) or sapg.off_policy_ratio < 0:
            raise ValueError("off_policy_ratio must be a non-negative integer")
        if sapg.entropy_coef_scale < 0:
            raise ValueError("entropy_coef_scale must be non-negative")
        if sapg.recurrent_experience_sharing not in ("reject", "reuse_state"):
            raise ValueError(
                "recurrent_experience_sharing must be 'reject' or 'reuse_state'"
            )

        if cfg.epo is not None:
            if sapg.num_conditionings < 2:
                raise ValueError("EPO requires at least two conditionings")
            if cfg.epo.evolution_frequency <= 0:
                raise ValueError("evolution_frequency must be positive")
            if not 0 < cfg.epo.evolution_kill_ratio < 1:
                raise ValueError(
                    "evolution_kill_ratio must be strictly between 0 and 1"
                )

    def __init__(
        self,
        experiment_dir: Path,
        ppo_config: PpoConfig,
        network_config: NetworkConfig,
        env: Any,
        observer: Optional[Any] = None,
    ):
        self.cfg = ppo_config

        ## A2CBase ##
        # multi-gpu/multi-node data
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        if self.cfg.multi_gpu:
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            dist.init_process_group(
                "nccl", rank=self.global_rank, world_size=self.world_size
            )

            self.device_name = "cuda:" + str(self.local_rank)
            self.cfg.device = self.device_name
            if self.global_rank != 0:
                self.cfg.print_stats = False
                self.cfg.lr_schedule = None

        self.env = env
        self.observer = observer if observer is not None else DefaultAlgoObserver()
        self.env_info = self.env.get_env_info()
        self.value_size = self.env_info.get("value_size", 1)
        self.observation_space = self.env_info["observation_space"]
        self.num_agents = self.env_info.get("agents", 1)
        self._validate_config()

        self.has_asymmetric_critic = self.cfg.asymmetric_critic is not None
        if self.has_asymmetric_critic:
            self.state_space = self.env_info.get("state_space", None)
            if isinstance(self.state_space, gym.spaces.Dict):
                raise NotImplementedError("Dict state space is not supported")
            else:
                self.state_shape = self.state_space.shape

        self.rnn_states: Optional[Tuple[torch.Tensor, ...]] = None

        # Setting learning rate scheduler
        if self.cfg.lr_schedule == "adaptive":
            assert self.cfg.kl_threshold is not None
            self.scheduler = schedulers.AdaptiveScheduler(
                kl_threshold=self.cfg.kl_threshold
            )

        elif self.cfg.lr_schedule == "linear":
            if self.cfg.max_epochs == -1 and self.cfg.max_frames == -1:
                print(
                    "Max epochs and max frames are not set. Linear learning rate schedule can't be used, switching to the contstant (identity) one."
                )
                self.scheduler = schedulers.IdentityScheduler()
            else:
                use_epochs = True
                max_steps = self.cfg.max_epochs

                if self.cfg.max_epochs == -1:
                    use_epochs = False
                    max_steps = self.cfg.max_frames

                self.scheduler = schedulers.LinearScheduler(
                    float(self.cfg.learning_rate),
                    max_steps=max_steps,
                    use_epochs=use_epochs,
                    apply_to_entropy=self.cfg.schedule_entropy,
                    start_entropy_coef=self.cfg.entropy_coef,
                )
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.rewards_shaper = DefaultRewardsShaper(self.cfg.reward_shaper)

        if isinstance(self.observation_space, gym.spaces.Dict):
            raise NotImplementedError("Dict observation space is not supported")
        else:
            self.obs_shape = self.observation_space.shape
            assert len(self.obs_shape) == 1, (
                f"Observation shape must be 1D, got {self.obs_shape}"
            )

        print("current training device:", self.device)
        self.game_rewards = torch_utils.AverageMeter(
            in_shape=self.value_size, max_size=self.cfg.games_to_track
        ).to(self.device)
        self.game_shaped_rewards = torch_utils.AverageMeter(
            in_shape=self.value_size, max_size=self.cfg.games_to_track
        ).to(self.device)
        self.game_lengths = torch_utils.AverageMeter(
            in_shape=1, max_size=self.cfg.games_to_track
        ).to(self.device)
        self.obs_dict: Optional[Dict[str, Any]] = None

        self.batch_size_envs = self.cfg.horizon_length * self.cfg.num_actors
        self.batch_size = self.batch_size_envs * self.num_agents

        # either minibatch_size_per_env or minibatch_size should be present in a config
        # if both are present, minibatch_size is used
        # otherwise minibatch_size_per_env is used minibatch_size_per_env is used to calculate minibatch_size
        self.minibatch_size = (
            self.cfg.minibatch_size
            if self.cfg.minibatch_size is not None
            else self.cfg.num_actors * self.cfg.minibatch_size_per_env
        )
        assert self.minibatch_size > 0

        assert self.batch_size % self.minibatch_size == 0, (
            f"batch_size ({self.batch_size}) must be divisible by minibatch_size ({self.minibatch_size})"
        )

        # For SAPG with experience sharing, the training batch is augmented:
        #   augmented_batch = batch + (num_repeat-1) * block_size_steps
        # where block_size_steps = horizon * (num_actors / M).
        if self.cfg.sapg is not None and self.cfg.sapg.use_others_experience:
            M = self.cfg.sapg.num_conditionings
            num_repeat = min(M, self.cfg.sapg.off_policy_ratio + 1)
            self.augmented_batch_size = self.batch_size + (num_repeat - 1) * (
                self.batch_size // M
            )
        else:
            self.augmented_batch_size = self.batch_size

        self.num_minibatches = self.augmented_batch_size // self.minibatch_size
        assert self.num_minibatches > 0, (
            f"{self.augmented_batch_size}, {self.minibatch_size}"
        )

        self.mixed_precision = (
            self.cfg.mixed_precision and torch.device(self.device).type == "cuda"
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.current_lr = float(self.cfg.learning_rate)
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.epoch_num = 0
        self.curr_frames = 0

        self.experiment_dir = experiment_dir
        self.nn_dir = self.experiment_dir / "nn"
        self.summaries_dir = self.experiment_dir / "summaries"

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.nn_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        self.current_entropy_coef = self.cfg.entropy_coef

        if self.global_rank == 0:
            writer = SummaryWriter(str(self.summaries_dir))
            self.writer = writer
        else:
            self.writer = None
        self.games_to_track = self.cfg.games_to_track
        self.observer.after_init(self)

        self.is_tensor_obses = False

        # SAPG/EPO: entropy intrinsic reward setup
        # uses_intrinsic_reward: controls per-block entropy in the LOSS (always True for SAPG)
        # uses_entropy_in_returns: controls whether entropy is also added to GAE returns.
        #   False (default): entropy only enters via the loss term (negative -entropy_loss).
        #   True: entropy is also added to shaped rewards before GAE (like an intrinsic bonus).
        #   rl_games stores intr_rewards=zeros for expl_type=entropy, so rl_games also uses
        #   False semantics (entropy only in loss).  Both default to this.  Set True to explore
        #   the alternative formulation.
        self.uses_intrinsic_reward = (
            self.cfg.sapg is not None and self.cfg.sapg.use_entropy_bonus
        )
        self.uses_entropy_in_returns = (
            self.uses_intrinsic_reward and self.cfg.sapg.entropy_in_returns
        )
        if self.uses_intrinsic_reward:
            M = self.cfg.sapg.num_conditionings
            assert self.cfg.num_actors % M == 0, (
                f"num_actors ({self.cfg.num_actors}) must be divisible by num_conditionings ({M})"
            )
            block_size = self.cfg.num_actors // M
            env_block_ids = (
                torch.arange(M).repeat_interleave(block_size).to(self.device)
            )
            # Per-env entropy coefficient, shape (num_actors,): used in GAE when entropy_in_returns=True
            self.intr_reward_coef = (
                torch.linspace(0.5, 0.0, M).to(self.device)[env_block_ids]
                * self.cfg.sapg.entropy_coef_scale
            )
            # Per-block entropy coefficient, shape (M,): used for per-sample lookup in loss
            self.intr_reward_coef_per_block = (
                torch.linspace(0.5, 0.0, M).to(self.device)
                * self.cfg.sapg.entropy_coef_scale
            )
        else:
            self.intr_reward_coef = None
            self.intr_reward_coef_per_block = None

        # EPO/SAPG: per-block episode reward tracking
        if self.cfg.sapg is not None:
            self.block_game_rewards = [
                torch_utils.AverageMeter(
                    in_shape=self.value_size, max_size=self.cfg.games_to_track
                ).to(self.device)
                for _ in range(self.cfg.sapg.num_conditionings)
            ]
        else:
            self.block_game_rewards = None

        # Conditioning idxs: integer index per env, appended to obs/states
        if self.cfg.sapg is not None:
            M = self.cfg.sapg.num_conditionings
            block_size = self.env.num_envs // M
            conditioning_idxs = (
                torch.arange(M)
                .repeat_interleave(block_size)
                .to(self.device)
                .reshape(self.env.num_envs, CONDITIONING_IDX_DIM)
            )
            self.conditioning_idxs = conditioning_idxs
        else:
            self.conditioning_idxs = None

        ## ContinuousA2CBase ##
        action_space = self.env_info["action_space"]
        self.actions_num = action_space.shape[0]

        self.actions_low = (
            torch.from_numpy(action_space.low.copy()).float().to(self.device)
        )
        self.actions_high = (
            torch.from_numpy(action_space.high.copy()).float().to(self.device)
        )

        ## A2CAgent ##
        self.model = models.ModelA2CContinuousLogStd(
            network_config=network_config,
            actions_num=self.actions_num,
            input_shape=self.obs_shape,
            normalize_value=self.cfg.normalize_value,
            normalize_input=self.cfg.normalize_input,
            value_size=self.env_info.get("value_size", 1),
            num_seqs=self.cfg.num_actors * self.num_agents,
            conditioning_dim=self.cfg.sapg.conditioning_dim
            if self.cfg.sapg is not None
            else None,
            num_conditionings=self.cfg.sapg.num_conditionings
            if self.cfg.sapg is not None
            else None,
        )

        self.model.to(self.device)

        self.states = None
        self.init_rnn_from_model(self.model)
        if self.is_rnn:
            if self.cfg.horizon_length % self.cfg.seq_length != 0:
                raise ValueError("horizon_length must be divisible by seq_length")
            if self.minibatch_size % self.cfg.seq_length != 0:
                raise ValueError("minibatch_size must be divisible by seq_length")
            if self.cfg.sapg is not None and self.cfg.sapg.use_others_experience:
                if self.cfg.sapg.recurrent_experience_sharing == "reject":
                    raise ValueError(
                        "SAPG off-policy experience sharing does not relabel saved recurrent "
                        "history. Set sapg.recurrent_experience_sharing='reuse_state' to "
                        "explicitly use the reference SAPG approximation, or disable sharing."
                    )
                print(
                    "WARNING: recurrent SAPG sharing is reusing saved LSTM/transformer "
                    "state under counterfactual conditioning (reference SAPG behavior)."
                )
        transformer_config = network_config.transformer
        if transformer_config is not None and not self.cfg.zero_rnn_on_done:
            raise ValueError(
                "transformer policies require zero_rnn_on_done=True so PPO replay "
                "cannot attend across outer episode boundaries"
            )
        if (
            transformer_config is not None
            and transformer_config.memory_mode == "segment"
            and self.cfg.seq_length != transformer_config.context_length
        ):
            raise ValueError(
                "segment transformer requires PPO seq_length to equal transformer context_length"
            )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            float(self.current_lr),
            eps=1e-08,
            weight_decay=self.cfg.weight_decay,
        )

        if self.has_asymmetric_critic:
            print("Adding Asymmetric Critic Network")
            assert self.cfg.asymmetric_critic is not None
            self.asymmetric_critic_net = asymmetric_critic.AsymmetricCritic(
                state_shape=self.state_shape,
                value_size=self.value_size,
                ppo_device=self.device,
                num_agents=self.num_agents,
                horizon_length=self.cfg.horizon_length,
                num_actors=self.cfg.num_actors,
                num_actions=self.actions_num,
                seq_length=self.cfg.seq_length,
                normalize_value=self.cfg.normalize_value,
                config=self.cfg.asymmetric_critic,
                writer=self.writer,
                max_epochs=self.cfg.max_epochs,
                multi_gpu=self.cfg.multi_gpu,
                zero_rnn_on_done=self.cfg.zero_rnn_on_done,
                training_batch_size=self.augmented_batch_size,
                conditioning_dim=self.cfg.sapg.conditioning_dim
                if self.cfg.sapg is not None
                else None,
                num_conditionings=self.cfg.sapg.num_conditionings
                if self.cfg.sapg is not None
                else None,
            ).to(self.device)

        self.dataset = datasets.PPODataset(
            batch_size=self.augmented_batch_size,
            minibatch_size=self.minibatch_size,
            is_rnn=self.is_rnn,
            device=self.device,
            seq_length=self.cfg.seq_length,
        )
        if self.cfg.normalize_value:
            self.value_mean_std = (
                self.asymmetric_critic_net.model.value_mean_std
                if self.has_asymmetric_critic
                else self.model.value_mean_std
            )

    def truncate_gradients_and_step(self) -> List[torch.Tensor]:
        if self.cfg.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(
                            param.grad.data
                        )
                        / self.world_size
                    )
                    offset += param.numel()
        else:
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))
            all_grads = torch.cat(all_grads_list)

        if self.cfg.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return all_grads

    def write_stats(
        self,
        total_time: float,
        epoch_num: int,
        step_time: float,
        play_time: float,
        update_time: float,
        a_losses: List[torch.Tensor],
        c_losses: List[torch.Tensor],
        entropies: List[torch.Tensor],
        kls: List[torch.Tensor],
        current_lr: float,
        lr_mul: float,
        frame: int,
        scaled_time: float,
        scaled_play_time: float,
        curr_frames: float,
        csigmas: Optional[List[torch.Tensor]] = None,
    ) -> None:
        if self.writer is None:
            print("writer is None, skipping writing stats")
            return

        # do we need scaled time?
        self.writer.add_scalar(
            tag="performance/step_inference_rl_update_fps",
            scalar_value=curr_frames / scaled_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_inference_fps",
            scalar_value=curr_frames / scaled_play_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_fps",
            scalar_value=curr_frames / step_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/rl_update_time",
            scalar_value=update_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_inference_time",
            scalar_value=play_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_time", scalar_value=step_time, global_step=frame
        )
        self.writer.add_scalar(
            tag="losses/a_loss",
            scalar_value=torch.mean(torch.stack(a_losses)).item(),
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="losses/c_loss",
            scalar_value=torch.mean(torch.stack(c_losses)).item(),
            global_step=frame,
        )

        self.writer.add_scalar(
            tag="losses/entropy",
            scalar_value=torch.mean(torch.stack(entropies)).item(),
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="info/current_lr", scalar_value=current_lr * lr_mul, global_step=frame
        )
        self.writer.add_scalar(
            tag="info/lr_mul", scalar_value=lr_mul, global_step=frame
        )
        self.writer.add_scalar(
            tag="info/e_clip", scalar_value=self.cfg.e_clip * lr_mul, global_step=frame
        )
        self.writer.add_scalar(
            tag="info/kl",
            scalar_value=torch.mean(torch.stack(kls)).item(),
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="info/epochs", scalar_value=epoch_num, global_step=frame
        )

        # EPO: action sigma logging
        if (
            self.cfg.epo is not None
            and self.cfg.epo.log_sigma
            and csigmas is not None
            and len(csigmas) > 0
        ):
            self.writer.add_scalar(
                tag="auxiliary_stats/sigma",
                scalar_value=torch.stack(csigmas).mean().item(),
                global_step=frame,
            )

        # EPO/SAPG: per-block reward logging
        if self.block_game_rewards is not None:
            for k, meter in enumerate(self.block_game_rewards):
                if meter.current_size > 0:
                    self.writer.add_scalar(
                        tag=f"block_rewards/block_{k}",
                        scalar_value=meter.get_mean()[0].item(),
                        global_step=frame,
                    )

        # SAPG: intrinsic reward logging
        if self.uses_intrinsic_reward and hasattr(self, "experience_buffer"):
            mb_intr_rewards = self.experience_buffer.tensor_dict.get(
                "intr_rewards", None
            )
            if mb_intr_rewards is not None:
                M = self.cfg.sapg.num_conditionings
                block_size = self.cfg.num_actors // M
                for k in range(M):
                    self.writer.add_scalar(
                        tag=f"intr_rewards/block_{k}",
                        scalar_value=mb_intr_rewards[
                            :, k * block_size : (k + 1) * block_size
                        ]
                        .mean()
                        .item(),
                        global_step=frame,
                    )
                mb_extr_rewards = self.experience_buffer.tensor_dict.get("rewards")
                if mb_extr_rewards is not None:
                    self.writer.add_scalar(
                        tag="intr_rewards/extr_rewards",
                        scalar_value=mb_extr_rewards.mean().item(),
                        global_step=frame,
                    )

    def set_eval(self) -> None:
        self.model.eval()

    def set_train(self) -> None:
        self.model.train()

    def update_lr(self, lr: float) -> None:
        if self.cfg.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, src=0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_action_values(self, obs_dict: Dict[str, Any]) -> dict:
        obs = obs_dict["obs"]
        self.model.eval()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.rnn_states,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_asymmetric_critic:
                states = obs_dict["states"]
                input_dict = {
                    "is_train": False,
                    "states": states,
                }
                value = self.get_asymmetric_critic_value(input_dict)
                res_dict["values"] = value
        return res_dict

    def get_values(
        self,
        obs_dict: Dict[str, Any],
        rnn_states: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.has_asymmetric_critic:
                states = obs_dict["states"]
                self.asymmetric_critic_net.eval()
                input_dict = {
                    "is_train": False,
                    "states": states,
                    "actions": None,
                    "is_done": self.dones,
                }
                value = self.get_asymmetric_critic_value(input_dict)
            else:
                self.model.eval()
                obs = obs_dict["obs"]
                input_dict = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": obs,
                    "rnn_states": rnn_states,
                }
                result = self.model(input_dict)
                value = result["values"]
            return value

    @property
    def device(self) -> Union[str, torch.device]:
        return self.cfg.device

    def init_tensors(self) -> None:
        ## A2CBase ##
        batch_size = self.num_agents * self.cfg.num_actors
        self.experience_buffer = ExperienceBuffer(
            env_info=self.env_info,
            num_actors=self.cfg.num_actors,
            horizon_length=self.cfg.horizon_length,
            has_asymmetric_critic=self.has_asymmetric_critic,
            device=self.device,
            extra_obs_dim=CONDITIONING_IDX_DIM if self.cfg.sapg is not None else None,
            extra_states_dim=CONDITIONING_IDX_DIM
            if self.has_asymmetric_critic and self.cfg.sapg is not None
            else None,
            aux_tensor_dict={"intr_rewards": (1,)}
            if self.uses_entropy_in_returns
            else None,
        )

        _val_shape = (self.cfg.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(
            current_rewards_shape, dtype=torch.float32, device=self.device
        )
        self.current_shaped_rewards = torch.zeros(
            current_rewards_shape, dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.memory_resets = self.dones.clone()

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.device) for s in self.rnn_states]

            total_agents = self.num_agents * self.cfg.num_actors
            num_seqs = self.cfg.horizon_length // self.cfg.seq_length
            self.mb_rnn_states = [
                torch.zeros(
                    (num_seqs, s.size()[0], total_agents, s.size()[2]),
                    dtype=torch.float32,
                    device=self.device,
                )
                for s in self.rnn_states
            ]

        ## ContinuousA2CBase ##
        self.update_list = ["actions", "neglogpacs", "values", "mus", "sigmas"]
        self.tensor_list = self.update_list + [
            "obses",
            "states",
            "dones",
            "memory_resets",
        ]

    def init_rnn_from_model(self, model) -> None:
        self.is_rnn = self.model.is_rnn()

    def _get_memory_resets(self, dones: torch.Tensor, infos: Any) -> torch.Tensor:
        """Return state-reset flags for the transition that just completed.

        Environments may preserve policy memory across a physical trial reset by
        returning ``infos["memory_reset"]``. Existing environments retain the
        historical behavior because the default is ``dones``.
        """
        if not isinstance(infos, dict) or "memory_reset" not in infos:
            return dones.byte()
        resets = infos["memory_reset"]
        if not isinstance(resets, torch.Tensor):
            resets = torch.as_tensor(resets, device=self.device)
        return resets.to(device=self.device, dtype=torch.uint8).reshape_as(dones)

    def cast_obs(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert obs.dtype != np.int8
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        return obs

    def obs_to_dict_of_tensors(self, obs) -> Dict[str, Any]:
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or "obs" not in obs:
            upd_obs = {"obs": upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = torch_utils.rescale_actions(
                self.actions_low, self.actions_high, clamped_actions
            )
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def env_step(self, actions: torch.Tensor) -> Tuple:
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.env.step(actions)

        obs_dict = self.obs_to_dict_of_tensors(obs)

        if self.conditioning_idxs is not None:
            obs_dict["obs"] = torch.cat(
                [obs_dict["obs"], self.conditioning_idxs], dim=1
            )
            if self.has_asymmetric_critic:
                obs_dict["states"] = torch.cat(
                    [obs_dict["states"], self.conditioning_idxs], dim=1
                )

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return (
                obs_dict,
                rewards.to(self.device),
                dones.to(self.device),
                infos,
            )
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return (
                obs_dict,
                torch.from_numpy(rewards).float().to(self.device),
                torch.from_numpy(dones).to(self.device),
                infos,
            )

    def env_reset(self):
        obs = self.env.reset()
        obs_dict = self.obs_to_dict_of_tensors(obs)
        if self.conditioning_idxs is not None:
            obs_dict["obs"] = torch.cat(
                [obs_dict["obs"], self.conditioning_idxs], dim=1
            )
            if self.has_asymmetric_critic:
                obs_dict["states"] = torch.cat(
                    [obs_dict["states"], self.conditioning_idxs], dim=1
                )
        return obs_dict

    def discount_values(
        self,
        fdones: torch.Tensor,
        last_extrinsic_values: torch.Tensor,
        mb_fdones: torch.Tensor,
        mb_extrinsic_values: torch.Tensor,
        mb_rewards: torch.Tensor,
    ) -> torch.Tensor:
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.cfg.horizon_length)):
            if t == self.cfg.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t + 1]
                nextvalues = mb_extrinsic_values[t + 1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = (
                mb_rewards[t]
                + self.cfg.gamma * nextvalues * nextnonterminal
                - mb_extrinsic_values[t]
            )
            mb_advs[t] = lastgaelam = (
                delta + self.cfg.gamma * self.cfg.tau * nextnonterminal * lastgaelam
            )
        return mb_advs

    def clear_stats(self) -> None:
        self.game_rewards.clear()
        self.game_shaped_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -100500
        if self.block_game_rewards is not None:
            for meter in self.block_game_rewards:
                meter.clear()

    def update_epoch(self) -> int:
        self.epoch_num += 1
        return self.epoch_num

    def train(self) -> Tuple[float, int]:
        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        wall_time_start = time.monotonic()
        self.obs_dict = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.cfg.multi_gpu:
            torch.cuda.set_device(self.local_rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            if self.has_asymmetric_critic:
                model_params.append(self.asymmetric_critic_net.state_dict())
            dist.broadcast_object_list(model_params, src=0)
            self.model.load_state_dict(model_params[0])
            if self.has_asymmetric_critic:
                self.asymmetric_critic_net.load_state_dict(model_params[1])

        while True:
            epoch_num = self.update_epoch()
            (
                step_time,
                play_time,
                update_time,
                sum_time,
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                current_lr,
                lr_mul,
                csigmas,
            ) = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # EPO: run evolutionary update if scheduled
            if (
                self.cfg.epo is not None
                and epoch_num % self.cfg.epo.evolution_frequency == 0
            ):
                self._run_evolutionary_update()

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = (
                    self.curr_frames * self.world_size
                    if self.cfg.multi_gpu
                    else self.curr_frames
                )
                self.frame += curr_frames

                print_statistics(
                    print_stats=self.cfg.print_stats,
                    curr_frames=curr_frames,
                    step_time=step_time,
                    step_inference_time=scaled_play_time,
                    total_time=scaled_time,
                    epoch_num=epoch_num,
                    max_epochs=self.cfg.max_epochs,
                    frame=frame,
                    max_frames=self.cfg.max_frames,
                )

                self.write_stats(
                    total_time=total_time,
                    epoch_num=epoch_num,
                    step_time=step_time,
                    play_time=play_time,
                    update_time=update_time,
                    a_losses=a_losses,
                    c_losses=c_losses,
                    entropies=entropies,
                    kls=kls,
                    current_lr=current_lr,
                    lr_mul=lr_mul,
                    frame=frame,
                    scaled_time=scaled_time,
                    scaled_play_time=scaled_play_time,
                    curr_frames=curr_frames,
                    csigmas=csigmas,
                )
                self.observer.after_print_stats(frame, epoch_num, total_time)

                if len(b_losses) > 0:
                    self.writer.add_scalar(
                        tag="losses/bounds_loss",
                        scalar_value=torch.mean(torch.stack(b_losses)).item(),
                        global_step=frame,
                    )

                if self.cfg.multi_gpu:
                    # Gather state from all gpus
                    state = self.get_full_state_weights()
                    state_list = [None] * self.world_size
                    dist.gather_object(state, state_list)
                    all_state_dict = {i: state_list[i] for i in range(self.world_size)}
                else:
                    all_state_dict = None

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = "rewards" if i == 0 else "rewards{0}".format(i)
                        self.writer.add_scalar(
                            tag=rewards_name + "/step".format(),
                            scalar_value=mean_rewards[i],
                            global_step=frame,
                        )
                        self.writer.add_scalar(
                            tag=rewards_name + "/iter".format(),
                            scalar_value=mean_rewards[i],
                            global_step=epoch_num,
                        )
                        self.writer.add_scalar(
                            tag=rewards_name + "/time".format(),
                            scalar_value=mean_rewards[i],
                            global_step=total_time,
                        )
                        self.writer.add_scalar(
                            tag="shaped_" + rewards_name + "/step".format(),
                            scalar_value=mean_shaped_rewards[i],
                            global_step=frame,
                        )
                        self.writer.add_scalar(
                            tag="shaped_" + rewards_name + "/iter".format(),
                            scalar_value=mean_shaped_rewards[i],
                            global_step=epoch_num,
                        )
                        self.writer.add_scalar(
                            tag="shaped_" + rewards_name + "/time".format(),
                            scalar_value=mean_shaped_rewards[i],
                            global_step=total_time,
                        )

                    self.writer.add_scalar(
                        tag="episode_lengths/step",
                        scalar_value=mean_lengths,
                        global_step=frame,
                    )
                    self.writer.add_scalar(
                        tag="episode_lengths/iter",
                        scalar_value=mean_lengths,
                        global_step=epoch_num,
                    )
                    self.writer.add_scalar(
                        tag="episode_lengths/time",
                        scalar_value=mean_lengths,
                        global_step=total_time,
                    )

                    if self.cfg.save_frequency > 0:
                        if epoch_num % self.cfg.save_frequency == 0:
                            self.save(
                                filename=(
                                    self.nn_dir
                                    / f"ep_{epoch_num}_rew_{mean_rewards[0]}.pth"
                                ),
                                override_state=all_state_dict,
                            )

                    if (
                        mean_rewards[0] > self.last_mean_rewards
                        and epoch_num >= self.cfg.save_best_after
                    ):
                        print("saving next best rewards: ", mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(
                            filename=(self.nn_dir / "best.pth"),
                            override_state=all_state_dict,
                        )

                if epoch_num >= self.cfg.max_epochs and self.cfg.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print(
                            "WARNING: Max epochs reached before any env terminated at least once"
                        )
                        mean_rewards = -np.inf

                    self.save(
                        filename=(
                            self.nn_dir
                            / f"last_ep_{epoch_num}_rew_{str(mean_rewards).replace('[', '_').replace(']', '_')}.pth"
                        ),
                        override_state=all_state_dict,
                    )
                    print("MAX EPOCHS NUM!")
                    should_exit = True

                if self.frame >= self.cfg.max_frames and self.cfg.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print(
                            "WARNING: Max frames reached before any env terminated at least once"
                        )
                        mean_rewards = -np.inf

                    self.save(
                        filename=(
                            self.nn_dir
                            / f"last_frame_{self.frame}_rew_{str(mean_rewards).replace('[', '_').replace(']', '_')}.pth"
                        ),
                        override_state=all_state_dict,
                    )
                    print("MAX FRAMES NUM!")
                    should_exit = True

                wall_time = time.monotonic() - wall_time_start
                if (
                    self.cfg.max_wall_time_seconds > 0
                    and wall_time >= self.cfg.max_wall_time_seconds
                ):
                    if self.game_rewards.current_size == 0:
                        mean_rewards = -np.inf
                    self.save(
                        filename=(
                            self.nn_dir
                            / f"last_walltime_{int(wall_time)}s_frame_{self.frame}_rew_"
                            f"{str(mean_rewards).replace('[', '_').replace(']', '_')}.pth"
                        ),
                        override_state=all_state_dict,
                    )
                    print("MAX WALL TIME!")
                    should_exit = True

                update_time = 0
            else:
                if self.cfg.multi_gpu:
                    state = self.get_full_state_weights()
                    dist.gather_object(state)

            if self.cfg.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, src=0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

    def prepare_dataset(self, batch_dict: Dict[str, Any]) -> None:
        obses = batch_dict["obses"]
        returns = batch_dict["returns"]
        dones = batch_dict["dones"]
        memory_resets = batch_dict.get("memory_resets", dones)
        values = batch_dict["values"]
        actions = batch_dict["actions"]
        neglogpacs = batch_dict["neglogpacs"]
        mus = batch_dict["mus"]
        sigmas = batch_dict["sigmas"]
        rnn_states = batch_dict.get("rnn_states", None)

        advantages = returns - values

        if self.cfg.normalize_value:
            if self.cfg.freeze_critic:
                self.value_mean_std.eval()
            else:
                self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, dim=1)

        if self.cfg.normalize_advantage:
            if self.cfg.multi_gpu:
                mean, var, _ = torch_utils.distributed_mean_var_count(
                    mean=advantages.mean(), var=advantages.var(), count=len(advantages)
                )
                std = torch.sqrt(var)
            else:
                mean = advantages.mean()
                std = advantages.std()
            advantages = (advantages - mean) / (std + 1e-8)

        dataset_dict = {}
        dataset_dict["old_values"] = values
        dataset_dict["old_logp_actions"] = neglogpacs
        dataset_dict["advantages"] = advantages
        dataset_dict["returns"] = returns
        dataset_dict["actions"] = actions
        dataset_dict["obs"] = obses
        dataset_dict["dones"] = dones
        dataset_dict["memory_resets"] = memory_resets
        dataset_dict["rnn_states"] = rnn_states
        dataset_dict["mu"] = mus
        dataset_dict["sigma"] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_asymmetric_critic:
            dataset_dict = {}
            dataset_dict["old_values"] = values
            dataset_dict["advantages"] = advantages
            dataset_dict["returns"] = returns
            dataset_dict["actions"] = actions
            dataset_dict["states"] = batch_dict["states"]
            dataset_dict["dones"] = dones
            self.asymmetric_critic_net.update_dataset(dataset_dict)

    def train_epoch(
        self,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        float,
        float,
        List[torch.Tensor],  # csigmas (EPO sigma tracking)
    ]:
        ## A2CBase ##
        self.env.set_train_info(self.frame, self)

        ## ContinuousA2CBase ##
        self.set_eval()
        play_time_start = time.perf_counter()
        with torch.no_grad():
            batch_dict, play_steps_extras = self.play_steps()

            if self.cfg.sapg is not None and self.cfg.sapg.use_others_experience:
                batch_dict = self.augment_batch_using_others_experience(
                    batch_dict=batch_dict, play_steps_extras=play_steps_extras
                )
            # Always shuffle for SAPG to balance block representation across minibatches.
            # rl_games does this unconditionally for mixed_expl; without it minibatches
            # are contiguous per block (e.g. mb0 = all block-0 samples, mb3 = all block-5)
            # which destabilises per-block gradient updates.
            if self.cfg.sapg is not None:
                batch_dict = shuffle_batch(
                    batch_dict=batch_dict,
                    horizon_length=self.cfg.horizon_length,
                    seq_length=self.cfg.seq_length if self.is_rnn else None,
                )

        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()

        self.set_train()
        self.curr_frames = batch_dict.pop("played_frames")
        self.prepare_dataset(batch_dict)
        if self.has_asymmetric_critic:
            self.train_asymmetric_critic()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        csigmas = []  # EPO: track mean action sigma per update

        for mini_ep in range(0, self.cfg.mini_epochs):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, current_lr, lr_mul, cmu, csigma, b_loss = (
                    self.train_actor_critic(self.dataset[i])
                )
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                csigmas.append(csigma.detach().mean())
                if self.cfg.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.cfg.schedule_type == "legacy":
                    av_kls = kl
                    if self.cfg.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.world_size
                    self.current_lr, self.current_entropy_coef = self.scheduler.update(
                        current_lr=self.current_lr,
                        entropy_coef=self.current_entropy_coef,
                        epoch=self.epoch_num,
                        frames=0,
                        kl_dist=av_kls.item(),
                    )
                    self.update_lr(self.current_lr)

            av_kls = torch.mean(torch.stack(ep_kls))
            if self.cfg.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size
            if self.cfg.schedule_type == "standard":
                self.current_lr, self.current_entropy_coef = self.scheduler.update(
                    current_lr=self.current_lr,
                    entropy_coef=self.current_entropy_coef,
                    epoch=self.epoch_num,
                    frames=0,
                    kl_dist=av_kls.item(),
                )
                self.update_lr(self.current_lr)

            kls.append(av_kls)
            if self.cfg.normalize_input:
                self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch

        update_time_end = time.perf_counter()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return (
            batch_dict["step_time"],
            play_time,
            update_time,
            total_time,
            a_losses,
            c_losses,
            b_losses,
            entropies,
            kls,
            current_lr,
            lr_mul,
            csigmas,
        )

    def train_actor_critic(
        self, input_dict: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        float,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute gradients needed to step the networks of the algorithm.

        Core algo logic is defined here

        Args:
            input_dict (:obj:`dict`): Algo inputs as a dict.

        """
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        returns_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]

        lr_mul = 1.0
        curr_e_clip = self.cfg.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        if self.is_rnn:
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.cfg.seq_length

            if self.cfg.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]
                batch_dict["memory_resets"] = input_dict["memory_resets"]

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropies = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            N = mu.shape[0]

            ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
            a_losses = torch.max(-surr1, -surr2)

            if self.has_asymmetric_critic:
                # The external asymmetric critic owns the value loss and optimizer.
                # This matches rl_games with central_value_config enabled.
                c_losses = torch.zeros(N, dtype=values.dtype, device=values.device)
            else:
                value_pred_clipped = value_preds_batch + (
                    values - value_preds_batch
                ).clamp(-curr_e_clip, curr_e_clip)
                value_losses = (values - returns_batch) ** 2
                value_losses_clipped = (value_pred_clipped - returns_batch) ** 2
                c_losses = torch.max(value_losses, value_losses_clipped).squeeze(dim=1)

            if self.cfg.bounds_loss_coef is not None:
                if self.cfg.bound_loss_type == "regularisation":
                    b_losses = (mu * mu).sum(dim=-1)
                elif self.cfg.bound_loss_type == "bound":
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_losses = (mu_loss_low + mu_loss_high).sum(dim=-1)
                else:
                    raise ValueError(
                        f"Unknown bound loss type {self.cfg.bound_loss_type}"
                    )
            else:
                b_losses = torch.zeros(N, device=self.device)

            assert a_losses.shape == (N,), (
                f"a_losses shape mismatch: {a_losses.shape} != {(N,)}"
            )
            assert c_losses.shape == (N,), (
                f"c_losses shape mismatch: {c_losses.shape} != {(N,)}"
            )
            assert entropies.shape == (N,), (
                f"entropies shape mismatch: {entropies.shape} != {(N,)}"
            )
            assert b_losses.shape == (N,), (
                f"b_losses shape mismatch: {b_losses.shape} != {(N,)}"
            )

            a_loss, c_loss, b_loss = (
                a_losses.mean(),
                c_losses.mean(),
                b_losses.mean(),
            )

            if self.uses_intrinsic_reward:
                # Per-sample entropy coefficient: look up from conditioning index in obs
                conditioning_idx = obs_batch[:, -CONDITIONING_IDX_DIM].long()  # (N,)
                entropy_coef_per_sample = self.intr_reward_coef_per_block[
                    conditioning_idx
                ]  # (N,)
                entropy = entropies.mean()
                entropy_loss = (entropy_coef_per_sample * entropies).mean()
            else:
                entropy = entropies.mean()
                entropy_loss = entropy * self.current_entropy_coef

            loss = (
                a_loss
                + 0.5 * c_loss * self.cfg.critic_coef
                - entropy_loss
                + (
                    b_loss * self.cfg.bounds_loss_coef
                    if self.cfg.bounds_loss_coef is not None
                    else 0.0
                )
            )
            if self.cfg.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.truncate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = True
            kl_dist = torch_utils.policy_kl(
                mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl
            )

        return (
            a_loss,
            c_loss,
            entropy,
            kl_dist,
            self.current_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
        )

    def get_asymmetric_critic_value(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
        return self.asymmetric_critic_net.get_value(obs_dict)

    def train_asymmetric_critic(self) -> float:
        return self.asymmetric_critic_net.train_net()

    def get_full_state_weights(self) -> Dict[str, Any]:
        state = self.get_weights()
        state["epoch"] = self.epoch_num
        state["frame"] = self.frame
        state["optimizer"] = self.optimizer.state_dict()

        if self.has_asymmetric_critic:
            state["assymetric_vf_nets"] = self.asymmetric_critic_net.state_dict()

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state["last_mean_rewards"] = self.last_mean_rewards

        if self.env is not None:
            env_state = self.env.get_env_state()
            state["env_state"] = env_state

        return state

    def set_full_state_weights(self, weights, set_epoch=True) -> None:
        self.set_weights(weights)
        if set_epoch:
            self.epoch_num = weights["epoch"]
            self.frame = weights["frame"]

        if self.has_asymmetric_critic:
            self.asymmetric_critic_net.load_state_dict(weights["assymetric_vf_nets"])

        self.optimizer.load_state_dict(weights["optimizer"])

        self.last_mean_rewards = weights.get("last_mean_rewards", -1000000000)

        if self.env is not None:
            env_state = weights.get("env_state", None)
            self.env.set_env_state(env_state)

    def get_weights(self) -> Dict[str, Any]:
        state = self.get_stats_weights(model_stats=True)
        state["model"] = self.model.state_dict()
        return state

    def get_stats_weights(self, model_stats=False) -> Dict[str, Any]:
        state = {}
        if self.mixed_precision:
            state["scaler"] = self.scaler.state_dict()
        if self.has_asymmetric_critic:
            state["central_val_stats"] = self.asymmetric_critic_net.get_stats_weights(
                model_stats
            )
        if model_stats:
            if self.cfg.normalize_input:
                state["running_mean_std"] = self.model.running_mean_std.state_dict()
            if self.cfg.normalize_value:
                state["reward_mean_std"] = self.model.value_mean_std.state_dict()

        return state

    def set_stats_weights(self, weights) -> None:
        if self.cfg.normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])
        if self.cfg.normalize_value and "reward_mean_std" in weights:
            self.model.value_mean_std.load_state_dict(weights["reward_mean_std"])
        if self.mixed_precision and "scaler" in weights:
            self.scaler.load_state_dict(weights["scaler"])

    def set_weights(self, weights) -> None:
        self.model.load_state_dict(weights["model"])
        self.set_stats_weights(weights)

    def play_steps(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        update_list = self.update_list
        step_time = 0.0

        store_step_rnn_states = (
            self.is_rnn
            and self.cfg.sapg is not None
            and self.cfg.sapg.use_others_experience
            and not self.has_asymmetric_critic
        )
        if self.is_rnn:
            mb_rnn_states = self.mb_rnn_states
        if store_step_rnn_states:
            rnn_state_buffer = [
                torch.zeros(
                    (self.cfg.horizon_length, *s.shape), dtype=s.dtype, device=s.device
                )
                for s in self.rnn_states
            ]

        for n in range(self.cfg.horizon_length):
            if self.is_rnn:
                if n % self.cfg.seq_length == 0:
                    for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                        mb_s[n // self.cfg.seq_length, :, :, :] = s

                if store_step_rnn_states:
                    for i, s in enumerate(self.rnn_states):
                        rnn_state_buffer[i][n, :, :, :] = s

                if self.has_asymmetric_critic:
                    self.asymmetric_critic_net.pre_step_rnn(n)

            res_dict = self.get_action_values(obs_dict=self.obs_dict)

            if self.is_rnn:
                self.rnn_states = res_dict["rnn_states"]

            self.experience_buffer.update_data("obses", n, self.obs_dict["obs"])
            self.experience_buffer.update_data("dones", n, self.dones.byte())
            self.experience_buffer.update_data(
                "memory_resets", n, self.memory_resets.byte()
            )

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_asymmetric_critic:
                self.experience_buffer.update_data("states", n, self.obs_dict["states"])

            if self.uses_entropy_in_returns:
                mu = res_dict["mus"]
                sigma = res_dict["sigmas"]
                entropy = (
                    torch.distributions.Normal(mu, sigma)
                    .entropy()
                    .sum(dim=-1, keepdim=True)
                )  # (N, 1)
                self.experience_buffer.update_data("intr_rewards", n, entropy)

            step_time_start = time.perf_counter()
            self.obs_dict, rewards, self.dones, infos = self.env_step(
                res_dict["actions"]
            )
            self.memory_resets = self._get_memory_resets(self.dones, infos)
            step_time_end = time.perf_counter()

            step_time += step_time_end - step_time_start

            shaped_rewards = self.rewards_shaper(rewards)

            if self.cfg.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.cfg.gamma
                    * res_dict["values"]
                    * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
                )

            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False).flatten()
            env_done_indices = all_done_indices[:: self.num_agents]
            self.observer.process_infos(infos, env_done_indices)

            memory_reset_indices = self.memory_resets.nonzero(as_tuple=False)
            if self.is_rnn and len(memory_reset_indices) > 0:
                if self.cfg.zero_rnn_on_done:
                    self.rnn_states = list(
                        self.model.reset_memory_states(
                            self.rnn_states, memory_reset_indices
                        )
                    )
            if len(all_done_indices) > 0:
                if self.has_asymmetric_critic:
                    self.asymmetric_critic_net.post_step_rnn(all_done_indices)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(
                self.current_shaped_rewards[env_done_indices]
            )
            self.game_lengths.update(
                self.current_lengths[env_done_indices].unsqueeze(-1)
            )

            if self.block_game_rewards is not None and len(env_done_indices) > 0:
                M = self.cfg.sapg.num_conditionings
                block_size = self.cfg.num_actors // M
                for k in range(M):
                    block_done = env_done_indices[
                        (env_done_indices >= k * block_size)
                        & (env_done_indices < (k + 1) * block_size)
                    ]
                    if len(block_done) > 0:
                        self.block_game_rewards[k].update(
                            self.current_rewards[block_done]
                        )

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = (
                self.current_shaped_rewards * not_dones.unsqueeze(1)
            )
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(
            obs_dict=self.obs_dict, rnn_states=self.rnn_states
        )

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()

        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_intr_rewards = self.experience_buffer.tensor_dict.get("intr_rewards", None)
        if mb_intr_rewards is not None:
            # intr_reward_coef: (num_actors,) → broadcast to (H, num_actors, 1)
            mb_total_rewards = (
                mb_rewards
                + self.intr_reward_coef.unsqueeze(0).unsqueeze(-1) * mb_intr_rewards
            )
        else:
            mb_total_rewards = mb_rewards
        mb_advs = self.discount_values(
            fdones=fdones,
            last_extrinsic_values=last_values,
            mb_fdones=mb_fdones,
            mb_extrinsic_values=mb_values,
            mb_rewards=mb_total_rewards,
        )
        mb_returns = mb_advs + mb_values
        batch_dict = self.experience_buffer.get_transformed_list(
            transform_op=swap_and_flatten01, tensor_list=self.tensor_list
        )

        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size
        batch_dict["step_time"] = step_time

        if self.is_rnn:
            states = []
            for mb_s in mb_rnn_states:
                t_size = mb_s.size()[0] * mb_s.size()[2]
                h_size = mb_s.size()[3]
                states.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))
            batch_dict["rnn_states"] = states

        extras = {
            "rewards": mb_rewards,
            "obs": self.experience_buffer.tensor_dict["obses"],
            "last_obs_dict": self.obs_dict,
            "states": self.experience_buffer.tensor_dict.get("states", None),
            "dones": mb_fdones,
            "last_dones": fdones,
            "rnn_states": rnn_state_buffer if store_step_rnn_states else None,
            "last_rnn_states": self.rnn_states,
            "mb_extr_rewards": mb_rewards,
            "mb_intr_rewards": mb_intr_rewards,  # SAPG: per-step entropy, or None
        }

        return batch_dict, extras

    def augment_batch_using_others_experience(
        self,
        batch_dict: Dict[str, Any],
        play_steps_extras: Dict[str, Any],
    ) -> Dict[str, Any]:
        new_batch_dict = {}
        num_blocks = self.cfg.sapg.num_conditionings
        block_size = self.env.num_envs // num_blocks

        num_repeat = min(num_blocks, int(self.cfg.sapg.off_policy_ratio) + 1)
        sampled_block_idxs = [0] + [
            int(x)
            for x in np.random.choice(
                range(1, num_blocks), size=num_repeat - 1, replace=False
            )
        ]
        if self.cfg.multi_gpu:
            dist.broadcast_object_list(sampled_block_idxs, src=0)

        for key, val in batch_dict.items():
            if key in ["played_frames", "step_time"]:
                new_batch_dict[key] = val
            elif key == "obses":
                intr_coef_embd = torch.cat(
                    [
                        torch.roll(
                            self.conditioning_idxs, shifts=block_size * i, dims=0
                        )
                        for i in sampled_block_idxs
                    ],
                    dim=0,
                )
                obses = torch.cat([val] * len(sampled_block_idxs), dim=0)
                obses[:, -CONDITIONING_IDX_DIM:] = intr_coef_embd.repeat_interleave(
                    self.cfg.horizon_length, dim=0
                )
                mask = torch.zeros(len(obses), dtype=torch.bool, device=obses.device)
                mask[len(val) :] = True
                obses = filter_leader(
                    val=obses,
                    orig_len=len(val),
                    sampled_block_idxs=sampled_block_idxs,
                    num_blocks=num_blocks,
                )
                mask = filter_leader(
                    val=mask,
                    orig_len=len(val),
                    sampled_block_idxs=sampled_block_idxs,
                    num_blocks=num_blocks,
                )
                new_batch_dict[key] = obses
                new_batch_dict["off_policy_mask"] = mask
            elif key == "states":
                if val is None:
                    new_batch_dict[key] = None
                elif self.conditioning_idxs is not None:
                    # Roll the conditioning idx in states for each sampled block
                    intr_coef_embd = torch.cat(
                        [
                            torch.roll(
                                self.conditioning_idxs,
                                shifts=block_size * i,
                                dims=0,
                            )
                            for i in sampled_block_idxs
                        ],
                        dim=0,
                    )
                    states = torch.cat([val] * len(sampled_block_idxs), dim=0)
                    states[:, -CONDITIONING_IDX_DIM:] = (
                        intr_coef_embd.repeat_interleave(self.cfg.horizon_length, dim=0)
                    )
                    new_batch_dict[key] = filter_leader(
                        val=states,
                        orig_len=len(val),
                        sampled_block_idxs=sampled_block_idxs,
                        num_blocks=num_blocks,
                    )
                else:
                    new_batch_dict[key] = filter_leader(
                        val=torch.cat([val] * len(sampled_block_idxs), dim=0),
                        orig_len=len(val),
                        sampled_block_idxs=sampled_block_idxs,
                        num_blocks=num_blocks,
                    )
            elif key in ["values", "returns"]:
                pass  # handled below
            elif key == "rnn_states":
                if val is not None:
                    augmented_states = []
                    for state in val:
                        sequences_per_block = state.shape[1] // num_blocks
                        parts = [state]
                        for block_idx in sampled_block_idxs[1:]:
                            start = (block_idx - 1) * sequences_per_block
                            end = block_idx * sequences_per_block
                            parts.append(state[:, start:end])
                        augmented_states.append(torch.cat(parts, dim=1))
                    new_batch_dict[key] = augmented_states
                else:
                    new_batch_dict[key] = None
            else:
                new_batch_dict[key] = filter_leader(
                    val=torch.cat([val] * len(sampled_block_idxs), dim=0),
                    orig_len=len(val),
                    sampled_block_idxs=sampled_block_idxs,
                    num_blocks=num_blocks,
                )

        new_returns_list = [batch_dict["returns"]]
        new_values_list = [batch_dict["values"]]

        for block_idx in sampled_block_idxs[1:]:
            mb_rewards = play_steps_extras["rewards"]
            mb_obs = play_steps_extras["obs"]
            last_obs_dict = play_steps_extras["last_obs_dict"]
            last_rnn_states = play_steps_extras["last_rnn_states"]
            mb_states = play_steps_extras["states"]
            mb_rnn_states = play_steps_extras["rnn_states"]

            rolled_conditioning = torch.roll(
                self.conditioning_idxs, shifts=block_size * block_idx, dims=0
            )
            mb_obs[:, :, -CONDITIONING_IDX_DIM:] = rolled_conditioning
            last_obs_dict["obs"][:, -CONDITIONING_IDX_DIM:] = rolled_conditioning

            # Also update states conditioning if using asymmetric critic
            if mb_states is not None and self.conditioning_idxs is not None:
                mb_states[:, :, -CONDITIONING_IDX_DIM:] = rolled_conditioning
            if (
                "states" in last_obs_dict
                and last_obs_dict["states"] is not None
                and self.conditioning_idxs is not None
            ):
                last_obs_dict["states"][:, -CONDITIONING_IDX_DIM:] = rolled_conditioning

            flattened_rnn_states = (
                [
                    rnn_s.transpose(0, 1).reshape(
                        rnn_s.transpose(0, 1).shape[0], -1, *rnn_s.shape[3:]
                    )
                    for rnn_s in mb_rnn_states
                ]
                if mb_rnn_states is not None
                else None
            )

            flattened_mb_obs = mb_obs.reshape(-1, *mb_obs.shape[2:])
            flattened_mb_states = (
                mb_states.reshape(-1, *mb_states.shape[2:])
                if mb_states is not None
                else None
            )

            CHUNK_SIZE = 8192  # Can't do this in one pass because OOM
            all_indices = torch.arange(flattened_mb_obs.shape[0], device=self.device)
            transformer_config = self.model.a2c_network.config.transformer
            if (
                transformer_config is not None
                and transformer_config.memory_mode == "segment"
                and flattened_rnn_states is not None
            ):
                # Segment inference requires a common phase within each vectorized
                # call. Flattening time and environments mixes phases, so group by
                # the saved global segment phase and scatter back afterward.
                phase = flattened_rnn_states[-1][0, :, 0].round().long()
                index_groups = [all_indices[phase == p] for p in phase.unique()]
            else:
                index_groups = [all_indices]

            mb_values = torch.empty(
                (len(flattened_mb_obs), self.value_size),
                dtype=batch_dict["values"].dtype,
                device=self.device,
            )
            for group in index_groups:
                for start_idx in range(0, len(group), CHUNK_SIZE):
                    indices = group[start_idx : start_idx + CHUNK_SIZE]
                    mb_values[indices] = self.get_values(
                        obs_dict={
                            "obs": flattened_mb_obs[indices],
                            "states": flattened_mb_states[indices]
                            if mb_states is not None
                            else None,
                        },
                        rnn_states=[s[:, indices] for s in flattened_rnn_states]
                        if flattened_rnn_states is not None
                        else None,
                    )
            last_values = self.get_values(
                obs_dict=last_obs_dict, rnn_states=last_rnn_states
            )

            mb_values = mb_values.reshape(*mb_obs.shape[:2], *mb_values.shape[1:])
            mb_values = torch.cat([mb_values, last_values.unsqueeze(0)], dim=0)

            mb_fdones = play_steps_extras["dones"]
            fdones = play_steps_extras["last_dones"]

            mb_fdones = torch.cat([mb_fdones, fdones.unsqueeze(0)], dim=0)
            mb_intr_rewards = play_steps_extras["mb_intr_rewards"]
            if mb_intr_rewards is not None:
                # Roll entropy coefficients to match this off-policy block's assignment
                rolled_intr_coef = torch.roll(
                    self.intr_reward_coef, shifts=block_size * block_idx, dims=0
                )  # (num_actors,)
                mb_returns = (
                    mb_rewards
                    + rolled_intr_coef.unsqueeze(0).unsqueeze(-1) * mb_intr_rewards
                    + self.cfg.gamma * mb_values[1:] * (1 - mb_fdones[1:]).unsqueeze(-1)
                )
            else:
                mb_returns = mb_rewards + self.cfg.gamma * mb_values[1:] * (
                    1 - mb_fdones[1:]
                ).unsqueeze(-1)

            new_returns_list.append(swap_and_flatten01(mb_returns))
            new_values_list.append(swap_and_flatten01(mb_values[:-1]))

        new_batch_dict["returns"] = filter_leader(
            val=torch.cat(new_returns_list, dim=0),
            orig_len=len(batch_dict["returns"]),
            sampled_block_idxs=sampled_block_idxs,
            num_blocks=num_blocks,
        )
        new_batch_dict["values"] = filter_leader(
            val=torch.cat(new_values_list, dim=0),
            orig_len=len(batch_dict["values"]),
            sampled_block_idxs=sampled_block_idxs,
            num_blocks=num_blocks,
        )

        # Reset obs / states conditioning back to original in play_steps_extras
        # (they were mutated in the per-block loop above)
        play_steps_extras["obs"][:, :, -CONDITIONING_IDX_DIM:] = self.conditioning_idxs
        play_steps_extras["last_obs_dict"]["obs"][:, -CONDITIONING_IDX_DIM:] = (
            self.conditioning_idxs
        )
        if (
            play_steps_extras["states"] is not None
            and self.conditioning_idxs is not None
        ):
            play_steps_extras["states"][:, :, -CONDITIONING_IDX_DIM:] = (
                self.conditioning_idxs
            )
        last_obs_dict = play_steps_extras["last_obs_dict"]
        if (
            "states" in last_obs_dict
            and last_obs_dict["states"] is not None
            and self.conditioning_idxs is not None
        ):
            last_obs_dict["states"][:, -CONDITIONING_IDX_DIM:] = self.conditioning_idxs

        return new_batch_dict

    def _run_evolutionary_update(self) -> None:
        """EPO: evolutionary update step.

        Ranks all blocks by mean episode reward. Kills the worst `kill_ratio`
        fraction. Replaces dead slots with pairwise-averaged conditioning
        embeddings of the survivors.
        """
        assert self.cfg.sapg is not None, (
            "_run_evolutionary_update requires sapg config"
        )
        assert self.cfg.epo is not None, "_run_evolutionary_update requires epo config"
        M = self.cfg.sapg.num_conditionings
        num_kill = max(1, min(M - 1, int(M * self.cfg.epo.evolution_kill_ratio)))
        num_survive = M - num_kill

        # Aggregate reward sums and counts so every distributed rank performs
        # exactly the same evolution. Do not kill unevaluated policies.
        reward_sums = torch.stack(
            [meter.mean[0] * meter.current_size for meter in self.block_game_rewards]
        ).to(self.device)
        reward_counts = torch.stack(
            [meter.current_size for meter in self.block_game_rewards]
        ).to(device=self.device, dtype=torch.float32)
        if self.cfg.multi_gpu:
            dist.all_reduce(reward_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(reward_counts, op=dist.ReduceOp.SUM)
        if (reward_counts == 0).any():
            if self.global_rank == 0:
                missing = (reward_counts == 0).nonzero(as_tuple=False).flatten()
                print(
                    "EPO evolution skipped: no completed episodes for blocks "
                    f"{missing.tolist()}"
                )
            return

        block_rewards = reward_sums / reward_counts
        sorted_idx = torch.argsort(block_rewards, descending=True)  # best first
        survivor_idx = sorted_idx[:num_survive]
        dead_idx = sorted_idx[num_survive:]

        # Recombine: pairwise averaging of survivors → replace dead slots
        network = self.model.a2c_network
        evolved_parameters = [network.conditioning]
        if network.fixed_sigma and network.sigma.ndim == 2:
            evolved_parameters.append(network.sigma)

        with torch.no_grad():
            for parameter in evolved_parameters:
                survivors = parameter[survivor_idx].clone()
                for i, dead in enumerate(dead_idx):
                    parent_a = survivors[i % num_survive]
                    parent_b = survivors[(i + 1) % num_survive]
                    parameter[dead] = (parent_a + parent_b) / 2.0

                # Adam momentum from a killed policy must not be applied to its
                # newly inherited parameters on the next optimizer step.
                optimizer_state = self.optimizer.state.get(parameter, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in optimizer_state:
                        optimizer_state[key][dead_idx] = 0

        # The active recurrent state was produced by the killed policy. Clear
        # only those environment blocks whose conditioning was replaced.
        if self.is_rnn and self.rnn_states is not None:
            dead_envs = torch.isin(
                self.conditioning_idxs[:, 0].long(), dead_idx
            ).nonzero(as_tuple=False)
            if len(dead_envs) > 0:
                self.rnn_states = list(
                    self.model.reset_memory_states(self.rnn_states, dead_envs)
                )

        # Reset per-block trackers so next evaluation reflects new embeddings
        for k in range(M):
            self.block_game_rewards[k].clear()

        print(
            f"EPO evolution: survivors={sorted_idx[:num_survive].tolist()}, "
            f"replaced={dead_idx.tolist()}"
        )

    def save(
        self, filename: Path, override_state: Optional[Dict[str, Any]] = None
    ) -> None:
        state = {self.global_rank: self.get_full_state_weights()}
        if override_state is not None:
            state = override_state
        torch_utils.save_checkpoint(filename=filename, state=state)

    def restore(self, filename: Path, set_epoch: bool = True) -> None:
        checkpoint = torch_utils.load_checkpoint(filename, map_location=self.device)
        # Training checkpoints are keyed by global rank so a distributed run
        # can restore each rank's optimizer/environment state. Plain model
        # dictionaries remain accepted for backward compatibility.
        if "model" not in checkpoint:
            rank = self.global_rank if self.global_rank in checkpoint else 0
            if rank not in checkpoint:
                raise KeyError(
                    f"checkpoint has no state for rank {rank}; keys={checkpoint.keys()}"
                )
            checkpoint = checkpoint[rank]
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)
        if self.is_rnn and self.rnn_states is not None:
            self.rnn_states = list(self.model.get_default_rnn_state())
            self.rnn_states = [state.to(self.device) for state in self.rnn_states]

    def override_sigma(self, sigma) -> None:
        net = self.model.a2c_network
        if hasattr(net, "sigma") and hasattr(net, "fixed_sigma"):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(sigma))
            else:
                print("Print cannot set new sigma because fixed_sigma is False")
        else:
            print("Print cannot set new sigma because sigma is not present")
