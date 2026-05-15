"""rl_games-compatible CNN+MLP+Gaussian network for the depth student.

Non-recurrent counterpart to ``depth_cnn_lstm.py``, matching the paper
"Perceptive Humanoid Parkour" (arXiv:2602.15827, Sec. III-D, Tab. VI):
"3-layer CNN and a 5-layer MLP with hidden sizes [2048, 1024, 512, 256, 128]".

Architecture:
    obs_dict['obs']: (B, image_flat + proprio + maybe block_id)
                              | split
    image (B, C, H, W) --> 3-conv CNN (stride 2, GroupNorm, ReLU) + GAP --+
                                                                          |--> concat --> 5-layer MLP --> mu_head / value_head
    proprio (B, P + block_id?) -------------------------------------------+

Notes:
- ``forward`` returns ``(mu, logstd, value, None)`` -- rl_games' model
  wrapper handles a ``None`` rnn-state for non-RNN networks.
- ``fixed_sigma`` only supports ``'fixed'`` (paper uses fixed sigma).
  SAPG's ``'coef_cond'`` branch is intentionally not ported -- it pairs
  with the SAPG intr-reward block, which the PPO path does not have.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder


class DepthCNNMLPBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__()
        self.params: dict | None = None

    def load(self, params: dict) -> None:
        self.params = params

    def build(self, name: str, **kwargs) -> "DepthCNNMLPBuilder.Network":
        return DepthCNNMLPBuilder.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params: dict, **kwargs) -> None:
            super().__init__()
            self.actions_num = int(kwargs["actions_num"])
            input_shape = tuple(kwargs["input_shape"])
            assert len(input_shape) == 1, (
                f"depth_cnn_mlp expects a flat obs vector; got {input_shape}"
            )
            obs_dim = int(input_shape[0])

            self.image_hw = tuple(params["image_hw"])
            self.image_channels = int(params.get("image_channels", 1))
            self.proprio_dim = int(params["proprio_dim"])
            self.has_block_id = bool(params.get("has_block_id", False))
            self.symmetric_critic = bool(params.get("symmetric_critic", False))

            self.fixed_sigma = params.get("fixed_sigma", "fixed")
            assert self.fixed_sigma == "fixed", (
                "depth_cnn_mlp only supports fixed_sigma='fixed' (no SAPG block-keyed sigma)"
            )

            cnn_channels = tuple(params.get("cnn_channels", (32, 64, 128)))
            # If set, project the GAP'd CNN feature down to `cnn_out_dim` before
            # concatenating with proprio. arXiv:2602.15827 Tab. VI uses 32.
            # Default 0 -> skip projection, feed cnn_channels[-1] directly.
            cnn_out_dim = int(params.get("cnn_out_dim", 0))
            mlp_hidden = tuple(
                params.get("mlp_hidden", (2048, 1024, 512, 256, 128))
            )
            log_sigma_init = float(params.get("log_sigma_init", -1.0))

            self._image_numel = (
                self.image_channels * self.image_hw[0] * self.image_hw[1]
            )
            self._block_id_numel = 1 if self.has_block_id else 0
            expected_obs_dim = (
                self._image_numel + self.proprio_dim + self._block_id_numel
            )
            assert obs_dim == expected_obs_dim, (
                f"obs_dim={obs_dim} but expected {expected_obs_dim} = "
                f"image_flat({self._image_numel}) + proprio({self.proprio_dim}) "
                f"+ block_id({self._block_id_numel}). "
                "Check image_hw / proprio_dim / has_block_id in the yaml."
            )

            # ---- CNN (paper: 3 layers, stride 2, GroupNorm, ReLU) + GAP ----
            cnn_layers = []
            in_c = self.image_channels
            for c in cnn_channels:
                cnn_layers += [
                    nn.Conv2d(in_c, c, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=min(8, c), num_channels=c),
                    nn.ReLU(inplace=True),
                ]
                in_c = c
            self.cnn = nn.Sequential(*cnn_layers)
            cnn_proj_layers = [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            ]
            if cnn_out_dim > 0:
                cnn_proj_layers += [
                    nn.Linear(cnn_channels[-1], cnn_out_dim),
                    nn.ELU(inplace=True),
                ]
                cnn_feat_dim = cnn_out_dim
            else:
                cnn_feat_dim = cnn_channels[-1]
            self.cnn_proj = nn.Sequential(*cnn_proj_layers)

            # ---- 5-layer MLP trunk (paper hidden sizes) ----
            trunk_in = cnn_feat_dim + self.proprio_dim + self._block_id_numel
            mlp_layers = []
            prev = trunk_in
            for h in mlp_hidden:
                mlp_layers += [nn.Linear(prev, h), nn.ELU(inplace=True)]
                prev = h
            self.trunk = nn.Sequential(*mlp_layers)
            trunk_out_dim = mlp_hidden[-1]

            # ---- mu head ----
            # Teacher emits raw mu (no Tanh) -- match that here so BC has no
            # action-range floor when the teacher saturates.
            self.mu_head = nn.Linear(trunk_out_dim, self.actions_num)

            # ---- fixed sigma ----
            self.log_sigma = nn.Parameter(
                torch.full(
                    (self.actions_num,), log_sigma_init, dtype=torch.float32
                )
            )

            # ---- optional symmetric value head ----
            if self.symmetric_critic:
                self.value_head = nn.Linear(trunk_out_dim, 1)
            else:
                self.value_head = None

        # ----- rl_games hooks -----

        def is_separate_critic(self) -> bool:
            return False

        def is_rnn(self) -> bool:
            return False

        # ----- forward -----

        def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            B = obs.shape[0]
            image = obs[:, : self._image_numel].view(
                B, self.image_channels, *self.image_hw
            )
            tail = obs[:, self._image_numel :]
            return image, tail

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            image, proprio_aug = self._split_obs(obs)

            cnn_feat = self.cnn_proj(self.cnn(image))           # (B, cnn_feat_dim)
            fused = torch.cat([cnn_feat, proprio_aug], dim=-1)  # (B, trunk_in)
            trunk_out = self.trunk(fused)                       # (B, trunk_out_dim)

            mu = self.mu_head(trunk_out)
            logstd = self.log_sigma.expand_as(mu)

            if self.value_head is not None:
                value = self.value_head(trunk_out)
            else:
                value = mu.new_zeros(mu.shape[0], 1)

            return mu, logstd, value, None
