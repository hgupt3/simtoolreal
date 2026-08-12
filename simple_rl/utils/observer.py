from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable

import numpy as np
import torch


def _flatten_dict(value: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(_flatten_dict(item, name))
        else:
            flattened[name] = item
    return flattened


def _as_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return torch.as_tensor(value)
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return torch.as_tensor(value)
    return None


class DefaultAlgoObserver:
    """Environment-info logger compatible with rl_games' observer lifecycle.

    Scalar info values are logged directly. Common vector evaluation metrics get
    mean/median/max summaries, ``episode`` entries are averaged, and
    ``episode_cumulative`` values are accumulated until each environment ends.
    """

    SUMMARY_VECTORS = {"successes", "closest_keypoint_max_dist", "discounted_reward"}

    def __init__(self) -> None:
        self.algo: Any = None
        self.writer: Any = None
        self.direct_info: Dict[str, float] = {}
        self.episode_values: Dict[str, list[float]] = defaultdict(list)
        self.episode_cumulative: Dict[str, torch.Tensor] = {}
        self.episode_cumulative_avg: Dict[str, deque[float]] = {}

    def after_init(self, algo: Any) -> None:
        self.algo = algo
        self.writer = algo.writer

    def process_infos(self, infos: Any, done_indices: Iterable[Any], **_: Any) -> None:
        if not isinstance(infos, dict):
            return

        episode = infos.get("episode")
        if isinstance(episode, dict):
            for key, value in _flatten_dict(episode).items():
                tensor = _as_tensor(value)
                if tensor is not None and tensor.numel() > 0:
                    self.episode_values[key].append(float(tensor.float().mean().item()))

        cumulative = infos.get("episode_cumulative")
        if isinstance(cumulative, dict):
            done_list = [int(i.item() if isinstance(i, torch.Tensor) else i) for i in done_indices]
            for key, value in _flatten_dict(cumulative).items():
                tensor = _as_tensor(value)
                if tensor is None or tensor.ndim == 0:
                    continue
                tensor = tensor.detach().float()
                if key not in self.episode_cumulative:
                    self.episode_cumulative[key] = torch.zeros_like(tensor)
                self.episode_cumulative[key] += tensor
                averages = self.episode_cumulative_avg.setdefault(
                    key, deque(maxlen=self.algo.games_to_track)
                )
                for index in done_list:
                    averages.append(float(self.episode_cumulative[key][index].item()))
                    self.episode_cumulative[key][index] = 0

        direct: Dict[str, float] = {}
        for key, value in _flatten_dict(infos).items():
            if key.startswith("episode/") or key.startswith("episode_cumulative/"):
                continue
            tensor = _as_tensor(value)
            if tensor is not None and tensor.numel() == 1:
                direct[key] = float(tensor.item())

        for tag in self.SUMMARY_VECTORS:
            tensor = _as_tensor(infos.get(tag))
            if tensor is not None and tensor.numel() > 0:
                values = tensor.float().reshape(-1)
                direct[tag] = float(values.mean().item())
                direct[f"{tag}_median"] = float(values.median().item())
                direct[f"{tag}_max"] = float(values.max().item())
            for key, value in infos.items():
                if key.startswith(f"{tag}_per_block"):
                    block_tensor = _as_tensor(value)
                    if block_tensor is not None and block_tensor.numel() > 0:
                        direct[key] = float(block_tensor.float().mean().item())

        objective = _as_tensor(infos.get("true_objective"))
        if objective is not None and objective.numel() > 0:
            objective = objective.float().reshape(-1)
            direct["true_objective_mean"] = float(objective.mean().item())
            direct["true_objective_max"] = float(objective.max().item())
        self.direct_info = direct

    def after_print_stats(self, frame: int, epoch_num: int, total_time: float) -> None:
        if self.writer is None:
            self.episode_values.clear()
            return
        for key, values in self.episode_values.items():
            if values:
                self.writer.add_scalar(f"Episode/{key}", float(np.mean(values)), frame)
        self.episode_values.clear()

        for key, values in self.episode_cumulative_avg.items():
            if values:
                self.writer.add_scalar(f"episode_cumulative/{key}", float(np.mean(values)), frame)
                self.writer.add_scalar(f"episode_cumulative_min/{key}_min", float(np.min(values)), frame)
                self.writer.add_scalar(f"episode_cumulative_max/{key}_max", float(np.max(values)), frame)

        for key, value in self.direct_info.items():
            self.writer.add_scalar(f"{key}/frame", value, frame)
            self.writer.add_scalar(f"{key}/iter", value, epoch_num)
            self.writer.add_scalar(f"{key}/time", value, total_time)
