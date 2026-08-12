from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn as nn

from simple_rl.utils.conditioning_utils import CONDITIONING_IDX_DIM
from simple_rl.utils.layers.recurrent import RnnWithDones
from simple_rl.utils.layers.transformer import StatefulTransformer, TransformerConfig


@dataclass
class MlpConfig:
    units: Sequence[int]


@dataclass
class RnnConfig:
    units: int
    layers: int
    name: Literal["lstm", "gru"]
    layer_norm: bool = False
    before_mlp: bool = False
    concat_input: bool = False
    concat_output: bool = False


@dataclass
class NetworkConfig:
    mlp: MlpConfig
    rnn: Optional[RnnConfig] = None
    transformer: Optional[TransformerConfig] = None
    separate_value_mlp: bool = False
    asymmetric_critic: bool = False
    sigma_init: float = 0.0


class Network(nn.Module):
    def __init__(
        self,
        config: NetworkConfig,
        actions_num: int,
        input_shape: Sequence[int],
        value_size: int = 1,
        num_seqs: int = 1,
        conditioning_dim: Optional[int] = None,
        num_conditionings: Optional[int] = None,
    ) -> None:
        super().__init__()
        # NOTE: input_shape should be given excluding conditioning dim or conditioning idx dim
        #       It should be (O,) not (O + 1,) or (O + C,)
        assert len(input_shape) == 1, f"Input shape must be 1D, got {input_shape}"

        self.config = config

        if self.asymmetric_critic:
            self.is_continuous = False
            self.fixed_sigma = False
        else:
            self.is_continuous = True
            self.fixed_sigma = True

        self.value_size = value_size
        self.num_seqs = num_seqs
        self.conditioning_dim = conditioning_dim
        self.num_conditionings = num_conditionings

        self.actor_mlp = nn.Sequential()
        self.critic_mlp = nn.Sequential()

        # Conditioning
        if conditioning_dim is not None and num_conditionings is not None:
            # Learnable conditioning parameters
            self.conditioning = nn.Parameter(
                torch.randn(
                    (num_conditionings, conditioning_dim),
                    dtype=torch.float32,
                    requires_grad=True,
                ),
                requires_grad=True,
            )
            input_shape = (input_shape[0] + conditioning_dim,)

        input_size = self._calc_input_size(input_shape)

        mlp_input_size = input_size
        if len(self.units) == 0:
            mlp_output_size = input_size
        else:
            mlp_output_size = self.units[-1]

        # rnn concat_input: rnn's new input will be its [original input, output from previous stage (if exists)]
        # rnn concat_output: rnn's new output will be its [original output, output from the previous stage (if exists)]
        if self.has_transformer:
            if self.has_rnn:
                raise ValueError("rnn and transformer are mutually exclusive")
            if self.separate_value_mlp:
                raise NotImplementedError(
                    "transformer policies use a shared backbone; separate_value_mlp is unsupported"
                )
            if self.asymmetric_critic:
                raise NotImplementedError(
                    "use the existing external asymmetric MLP critic with a transformer policy"
                )
            if len(self.units) == 0:
                raise ValueError("transformer requires a non-empty mlp token embedding")
            out_size = mlp_output_size
        elif self.has_rnn:
            if not self.is_rnn_before_mlp:
                # mlp -> rnn

                # rnn(mlp(x))
                rnn_in_size = mlp_output_size

                if self.rnn_concat_input:
                    # rnn([mlp(x), x])
                    rnn_in_size += input_size

                # out = rnn(...)
                out_size = self.rnn_units

                if self.rnn_concat_output:
                    # out = rnn(...) + mlp(x)
                    out_size += mlp_output_size
            else:
                # rnn -> mlp

                # rnn(x)
                rnn_in_size = input_size

                # mlp(rnn(x))
                mlp_input_size = self.rnn_units

                if self.rnn_concat_output:
                    # mlp(rnn(x) + x)
                    mlp_input_size += input_size

                # out = mlp(...)
                out_size = mlp_output_size

            if self.separate_value_mlp:
                self.a_rnn = self._build_rnn(
                    self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                )
                self.c_rnn = self._build_rnn(
                    self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                )
                if self.rnn_ln:
                    self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                    self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
            else:
                self.rnn = self._build_rnn(
                    self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                )
                if self.rnn_ln:
                    self.layer_norm = torch.nn.LayerNorm(self.rnn_units)
        else:
            out_size = mlp_output_size

        self.actor_mlp = self._build_mlp(input_size=mlp_input_size, units=self.units)
        if self.separate_value_mlp:
            self.critic_mlp = self._build_mlp(
                input_size=mlp_input_size, units=self.units
            )

        if self.has_transformer:
            assert self.config.transformer is not None
            self.transformer = StatefulTransformer(
                config=self.config.transformer,
                d_model=mlp_output_size,
                num_seqs=self.num_seqs,
            )
            self.actor_head, actor_output_size = self._build_head(
                mlp_output_size, self.config.transformer.actor_head_units
            )
            self.critic_head, critic_output_size = self._build_head(
                mlp_output_size, self.config.transformer.critic_head_units
            )
        else:
            self.actor_head = nn.Identity()
            self.critic_head = nn.Identity()
            actor_output_size = critic_output_size = out_size

        self.value = torch.nn.Linear(critic_output_size, self.value_size)
        self.value_act = nn.Identity()

        if self.is_continuous:
            self.mu = torch.nn.Linear(actor_output_size, actions_num)
            self.mu_act = nn.Identity()
            self.sigma_act = nn.Identity()

            if self.fixed_sigma:
                # When SAPG conditioning is active, give each block its own sigma
                # vector (shape M × actions) — matches rl_games `coef_cond` behaviour.
                # This lets each block learn a different exploration width.
                # Without conditioning, use a single shared sigma (shape actions,).
                if num_conditionings is not None:
                    self.sigma = nn.Parameter(
                        torch.zeros(
                            num_conditionings,
                            actions_num,
                            requires_grad=True,
                            dtype=torch.float32,
                        ),
                        requires_grad=True,
                    )
                else:
                    self.sigma = nn.Parameter(
                        torch.zeros(
                            actions_num, requires_grad=True, dtype=torch.float32
                        ),
                        requires_grad=True,
                    )
            else:
                self.sigma = torch.nn.Linear(out_size, actions_num)

        mlp_init = nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        if self.is_continuous:
            mu_init = nn.Identity()
            sigma_init = partial(nn.init.constant_, val=self.config.sigma_init)
            mu_init(self.mu.weight)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

    def forward(self, obs_dict: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        obs = obs_dict["obs"]

        # Concatenate conditioning parameters to the observation
        if self.conditioning_dim is not None and self.num_conditionings is not None:
            # Observation is (N, O+1)
            normal_obs = obs[:, :-CONDITIONING_IDX_DIM]
            N = normal_obs.shape[0]

            conditioning_idxs = obs[:, -CONDITIONING_IDX_DIM:].long()
            assert (
                conditioning_idxs.min() >= 0
                and conditioning_idxs.max() < self.num_conditionings
            ), (
                f"Conditioning index out of bounds: {conditioning_idxs.min()}, {conditioning_idxs.max()}, {self.num_conditionings}"
            )
            assert conditioning_idxs.shape == (N, CONDITIONING_IDX_DIM), (
                f"Conditioning index shape mismatch: {conditioning_idxs.shape} != {(N, CONDITIONING_IDX_DIM)}"
            )
            conditioning_idxs = conditioning_idxs.squeeze(dim=1)

            conditioning = self.conditioning[conditioning_idxs]
            assert conditioning.shape == (N, self.conditioning_dim), (
                f"Conditioning shape mismatch: {conditioning.shape} != {(N, self.conditioning_dim)}"
            )

            # New observation is (N, O + C)
            obs = torch.cat([normal_obs, conditioning], dim=1)

        states = obs_dict.get("rnn_states", None)
        dones = obs_dict.get("memory_resets", obs_dict.get("dones", None))

        if self.has_transformer:
            if states is None:
                raise ValueError("transformer forward requires memory states")
            out = self.actor_mlp(obs.flatten(1))
            out, states = self.transformer(
                embeddings=out,
                states=states,
                is_train=obs_dict.get("is_train", False),
                seq_length=obs_dict.get("seq_length", 1),
                memory_resets=obs_dict.get("memory_resets", dones),
            )
            actor_out = self.actor_head(out)
            critic_out = self.critic_head(out)
            value = self.value_act(self.value(critic_out))
            mu = self.mu_act(self.mu(actor_out))
            if self.fixed_sigma:
                sigma_param = (
                    self.sigma[conditioning_idxs]
                    if self.num_conditionings is not None
                    else self.sigma
                )
                sigma = mu * 0.0 + self.sigma_act(sigma_param)
            else:
                sigma = self.sigma_act(self.sigma(actor_out))
            return mu, sigma, value, states

        if self.separate_value_mlp:
            a_out = c_out = obs
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            c_out = c_out.contiguous().view(c_out.size(0), -1)

            if self.has_rnn:
                seq_length = obs_dict.get("seq_length", 1)

                a_in = a_out
                c_in = c_out
                if not self.is_rnn_before_mlp:
                    # mlp -> rnn

                    # rnn(mlp(x))
                    a_out = self.actor_mlp(a_in)
                    c_out = self.critic_mlp(c_in)

                    a_mlp_out = a_out
                    c_mlp_out = c_out

                    if self.rnn_concat_input:
                        # rnn([mlp(x), x])
                        a_out = torch.cat([a_out, a_in], dim=1)
                        c_out = torch.cat([c_out, c_in], dim=1)

                batch_size = a_out.size()[0]
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1)
                c_out = c_out.reshape(num_seqs, seq_length, -1)

                a_out = a_out.transpose(0, 1)
                c_out = c_out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)

                if len(states) == 2:
                    a_states = states[0]
                    c_states = states[1]
                else:
                    a_states = states[:2]
                    c_states = states[2:]
                a_out, a_states = self.a_rnn(a_out, a_states, dones)
                c_out, c_states = self.c_rnn(c_out, c_states, dones)

                a_out = a_out.transpose(0, 1)
                c_out = c_out.transpose(0, 1)
                a_out = a_out.contiguous().reshape(
                    a_out.size()[0] * a_out.size()[1], -1
                )
                c_out = c_out.contiguous().reshape(
                    c_out.size()[0] * c_out.size()[1], -1
                )

                if self.rnn_ln:
                    a_out = self.a_layer_norm(a_out)
                    c_out = self.c_layer_norm(c_out)

                if type(a_states) is not tuple:
                    a_states = (a_states,)
                    c_states = (c_states,)
                states = a_states + c_states

                if self.rnn_concat_output and not self.is_rnn_before_mlp:
                    # mlp -> rnn

                    # out = rnn(...) + mlp(x)
                    a_out = torch.cat([a_out, a_mlp_out], dim=1)
                    c_out = torch.cat([c_out, c_mlp_out], dim=1)
                elif self.rnn_concat_output and self.is_rnn_before_mlp:
                    # rnn -> mlp

                    # out = mlp(rnn(x) + x)
                    a_out = torch.cat([a_out, a_in], dim=1)
                    c_out = torch.cat([c_out, c_in], dim=1)
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)

            else:
                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)

            value = self.value_act(self.value(c_out))

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.fixed_sigma:
                    if self.num_conditionings is not None:
                        # Per-block sigma: index by the conditioning idx in the obs.
                        # obs still has the original (O+1) shape here; conditioning_idxs
                        # was extracted above and is still in scope.
                        sigma = mu * 0.0 + self.sigma_act(self.sigma[conditioning_idxs])
                    else:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma, value, states
        else:
            out = obs
            out = out.flatten(1)

            if self.has_rnn:
                seq_length = obs_dict.get("seq_length", 1)

                in_ = out
                if not self.is_rnn_before_mlp:
                    # mlp -> rnn

                    # rnn(mlp(x))
                    out = self.actor_mlp(out)
                    mlp_out = out

                    if self.rnn_concat_input:
                        # rnn([mlp(x), x])
                        out = torch.cat([out, in_], dim=1)

                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)

                if len(states) == 1:
                    states = states[0]

                out = out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)
                out, states = self.rnn(out, states, dones)
                out = out.transpose(0, 1)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)
                if self.rnn_concat_output and not self.is_rnn_before_mlp:
                    # mlp -> rnn

                    # out = rnn(...) + mlp(x)
                    out = torch.cat([out, mlp_out], dim=1)
                elif self.rnn_concat_output and self.is_rnn_before_mlp:
                    # rnn -> mlp

                    # out = mlp(rnn(x) + x)
                    out = torch.cat([out, in_], dim=1)
                    out = self.actor_mlp(out)
                elif not self.rnn_concat_output and self.is_rnn_before_mlp:
                    # rnn -> mlp

                    # out = mlp(rnn(x))
                    out = self.actor_mlp(out)
                if type(states) is not tuple:
                    states = (states,)
            else:
                out = self.actor_mlp(out)
            value = self.value_act(self.value(out))

            if self.asymmetric_critic:
                return value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    if self.num_conditionings is not None:
                        sigma = self.sigma_act(self.sigma[conditioning_idxs])
                    else:
                        sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu * 0 + sigma, value, states

    def is_separate_critic(self) -> bool:
        return self.separate_value_mlp

    def is_rnn(self) -> bool:
        # Kept for compatibility with the existing PPO sequence-state plumbing.
        return self.has_rnn or self.has_transformer

    def get_default_rnn_state(self) -> Optional[Tuple[torch.Tensor, ...]]:
        if self.has_transformer:
            return self.transformer.get_default_state()
        if not self.has_rnn:
            return None
        num_layers = self.rnn_layers
        rnn_units = self.rnn_units
        if self.rnn_name == "lstm":
            if self.separate_value_mlp:
                return (
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                )
            else:
                return (
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                )
        else:
            if self.separate_value_mlp:
                return (
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                )
            else:
                return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)

    @property
    def separate_value_mlp(self) -> bool:
        return self.config.separate_value_mlp

    @property
    def units(self) -> Sequence[int]:
        return self.config.mlp.units

    @property
    def has_rnn(self) -> bool:
        return self.config.rnn is not None

    @property
    def has_transformer(self) -> bool:
        return self.config.transformer is not None

    @property
    def asymmetric_critic(self) -> bool:
        return self.config.asymmetric_critic

    @property
    def rnn_units(self) -> int:
        return self.config.rnn.units

    @property
    def rnn_layers(self) -> int:
        return self.config.rnn.layers

    @property
    def rnn_name(self) -> Literal["lstm", "gru"]:
        return self.config.rnn.name

    @property
    def rnn_ln(self) -> bool:
        return self.config.rnn.layer_norm

    @property
    def is_rnn_before_mlp(self) -> bool:
        return self.config.rnn.before_mlp

    @property
    def rnn_concat_input(self) -> bool:
        return self.config.rnn.concat_input

    @property
    def rnn_concat_output(self) -> bool:
        return self.config.rnn.concat_output

    def get_value_layer(self) -> nn.Module:
        return self.value

    def reset_memory_states(
        self, states: Sequence[torch.Tensor], indices: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        indices = indices.reshape(-1)
        if self.has_transformer:
            return self.transformer.reset_state(states, indices)
        reset_states = list(states)
        for state in reset_states:
            state[:, indices, :] = 0
        return tuple(reset_states)

    def _calc_input_size(self, input_shape: Sequence[int]) -> int:
        assert len(input_shape) == 1, f"Input shape must be 1D, got {input_shape}"
        return input_shape[0]

    def _build_rnn(
        self, name: Literal["lstm", "gru"], input: int, units: int, layers: int
    ) -> RnnWithDones:
        if name == "lstm":
            return RnnWithDones(
                rnn_layer=torch.nn.LSTM(
                    input_size=input, hidden_size=units, num_layers=layers
                ),
            )
        if name == "gru":
            return RnnWithDones(
                rnn_layer=torch.nn.GRU(
                    input_size=input, hidden_size=units, num_layers=layers
                ),
            )
        raise ValueError(f"Unknown rnn type: {name}")

    def _build_mlp(self, input_size: int, units: Sequence[int]) -> nn.Sequential:
        in_size = input_size
        layers = []
        for unit in units:
            layers.append(torch.nn.Linear(in_size, unit))
            layers.append(torch.nn.ELU())
            in_size = unit

        return nn.Sequential(*layers)

    def _build_head(
        self, input_size: int, units: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        if len(units) == 0:
            return nn.Sequential(), input_size
        return self._build_mlp(input_size, units), units[-1]
