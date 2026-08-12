from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from simple_rl.utils.conditioning_utils import CONDITIONING_IDX_DIM
from simple_rl.utils.network import Network, NetworkConfig
from simple_rl.utils.running_mean_std import RunningMeanStd


class BaseModel(nn.Module):
    def __init__(
        self,
        network_config: NetworkConfig,
        actions_num: int,
        input_shape: Sequence[int],
        normalize_value: bool,
        normalize_input: bool,
        value_size: int,
        num_seqs: int = 1,
        conditioning_dim: Optional[int] = None,
        num_conditionings: Optional[int] = None,
    ) -> None:
        nn.Module.__init__(self)

        # NOTE: input_shape should be given excluding conditioning dim or conditioning idx dim
        #       It should be (O,) not (O + 1,) or (O + C,)
        assert len(input_shape) == 1, (
            f"Currently only support 1D input shape, input_shape: {input_shape}"
        )

        self.a2c_network = Network(
            config=network_config,
            actions_num=actions_num,
            input_shape=input_shape,
            value_size=value_size,
            num_seqs=num_seqs,
            conditioning_dim=conditioning_dim,
            num_conditionings=num_conditionings,
        )
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input
        self.input_shape = input_shape
        self.conditioning_dim = conditioning_dim
        self.num_conditionings = num_conditionings

        if normalize_value:
            self.value_mean_std = RunningMeanStd((value_size,))
        if normalize_input:
            self.running_mean_std = RunningMeanStd(input_shape)

    def forward(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclass must implement forward method")

    def is_rnn(self):
        return self.a2c_network.is_rnn()

    def get_value_layer(self):
        return self.a2c_network.get_value_layer()

    def get_default_rnn_state(self):
        return self.a2c_network.get_default_rnn_state()

    def validate_obs_shape(self, observation: torch.Tensor) -> None:
        assert observation.ndim == 2, f"Observation must be 2D, got {observation.ndim}"
        _N, D = observation.shape

        expected_obs_dim = self.input_shape[0]
        if self.conditioning_dim is not None and self.num_conditionings is not None:
            expected_obs_dim += CONDITIONING_IDX_DIM

        assert D == expected_obs_dim, (
            f"Observation shape mismatch: {D} != {expected_obs_dim}"
        )

    def reset_memory_states(self, states, indices):
        return self.a2c_network.reset_memory_states(states, indices)

    def norm_obs(self, observation: torch.Tensor) -> torch.Tensor:
        self.validate_obs_shape(observation)

        if not self.normalize_input:
            return observation

        with torch.no_grad():
            if self.conditioning_dim is None or self.num_conditionings is None:
                return self.running_mean_std(observation)

            # Assume conditioning idx is at the end of the observation
            # Do not normalize the conditioning idx
            normal_obs, conditioning_idx = (
                observation[:, :-CONDITIONING_IDX_DIM],
                observation[:, -CONDITIONING_IDX_DIM:],
            )
            return torch.cat(
                [self.running_mean_std(normal_obs), conditioning_idx], dim=1
            )

    def denorm_value(self, value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return (
                self.value_mean_std(value, denorm=True)
                if self.normalize_value
                else value
            )


class ModelA2CContinuousLogStd(BaseModel):
    def forward(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_obs_shape(input_dict["obs"])

        is_train = input_dict.get("is_train", True)
        prev_actions = input_dict.get("prev_actions", None)
        input_dict["obs"] = self.norm_obs(input_dict["obs"])
        mu, logstd, value, states = self.a2c_network(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma, validate_args=False)
        if is_train:
            entropy = distr.entropy().sum(dim=-1)
            prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
            result = {
                "prev_neglogp": prev_neglogp.reshape(-1),
                "values": value,
                "entropy": entropy,
                "rnn_states": states,
                "mus": mu,
                "sigmas": sigma,
            }
            return result
        else:
            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, sigma, logstd)
            result = {
                "neglogpacs": neglogp.reshape(-1),
                "values": self.denorm_value(value),
                "actions": selected_action,
                "rnn_states": states,
                "mus": mu,
                "sigmas": sigma,
            }
            return result

    def neglogp(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        logstd: torch.Tensor,
    ) -> torch.Tensor:
        return (
            0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
            + logstd.sum(dim=-1)
        )


class ModelAsymmetricCritic(BaseModel):
    def forward(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_obs_shape(input_dict["obs"])

        is_train = input_dict.get("is_train", True)
        _prev_actions = input_dict.get("prev_actions", None)
        input_dict["obs"] = self.norm_obs(input_dict["obs"])
        value, states = self.a2c_network(input_dict)
        if not is_train:
            value = self.denorm_value(value)

        result = {"values": value, "rnn_states": states}
        return result
