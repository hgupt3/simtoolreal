"""Validated PCA/eigengrasp decoding for SimToolReal hand targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch


Normalization = Literal["clamp", "gauge"]


@dataclass(frozen=True)
class EigengraspArtifact:
    artifact_id: str
    source_corpora: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    mean: torch.Tensor
    components: torch.Tensor
    coefficient_low: torch.Tensor
    coefficient_high: torch.Tensor
    cumulative_variance: torch.Tensor

    @property
    def joint_dim(self) -> int:
        return len(self.joint_names)


def load_eigengrasp_artifact(path: str | Path) -> EigengraspArtifact:
    """Load and structurally validate the vendored Action-Bench JSON artifact."""

    artifact_path = Path(path)
    with artifact_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)

    if payload.get("schema") != "action-bench.hand-lab-pca-artifact":
        raise ValueError("unsupported eigengrasp artifact schema")
    if payload.get("version") != 1:
        raise ValueError("unsupported eigengrasp artifact version")

    names = tuple(payload["joint_names"])
    count = len(names)
    if count == 0 or len(set(names)) != count:
        raise ValueError("eigengrasp joint names must be nonempty and unique")

    limits = torch.as_tensor(payload["joint_limits"], dtype=torch.float32)
    mean = torch.as_tensor(payload["mean"], dtype=torch.float32)
    components = torch.as_tensor(payload["components"], dtype=torch.float32)
    coefficient_low = torch.as_tensor(
        payload["coefficient_low"], dtype=torch.float32
    )
    coefficient_high = torch.as_tensor(
        payload["coefficient_high"], dtype=torch.float32
    )
    cumulative_variance = torch.as_tensor(
        payload["cumulative_variance"], dtype=torch.float32
    )
    expected_vector = (count,)
    if limits.shape != (count, 2):
        raise ValueError("eigengrasp joint_limits have the wrong shape")
    if mean.shape != expected_vector:
        raise ValueError("eigengrasp mean has the wrong shape")
    if components.shape != (count, count):
        raise ValueError("eigengrasp artifact must contain a full-rank square basis")
    for name, tensor in (
        ("coefficient_low", coefficient_low),
        ("coefficient_high", coefficient_high),
        ("cumulative_variance", cumulative_variance),
    ):
        if tensor.shape != expected_vector:
            raise ValueError(f"eigengrasp {name} has the wrong shape")
    tensors = (
        limits,
        mean,
        components,
        coefficient_low,
        coefficient_high,
        cumulative_variance,
    )
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("eigengrasp artifact contains non-finite values")
    lower, upper = limits.unbind(dim=-1)
    if bool(torch.any(lower >= upper)):
        raise ValueError("eigengrasp joint limits are invalid")
    if bool(torch.any((mean <= lower) | (mean >= upper))):
        raise ValueError("eigengrasp mean must lie strictly inside its joint box")
    if bool(torch.any(coefficient_low >= 0.0)) or bool(
        torch.any(coefficient_high <= 0.0)
    ):
        raise ValueError("absolute eigengrasp coordinates must have two-sided bounds")
    gram = components.to(torch.float64) @ components.to(torch.float64).T
    if not torch.allclose(
        gram, torch.eye(count, dtype=torch.float64), atol=2.0e-5, rtol=2.0e-5
    ):
        raise ValueError("eigengrasp components are not orthonormal")
    if bool(torch.any(cumulative_variance[1:] < cumulative_variance[:-1])) or not bool(
        torch.isclose(cumulative_variance[-1], torch.tensor(1.0), atol=1.0e-5)
    ):
        raise ValueError("eigengrasp cumulative variance is invalid")

    return EigengraspArtifact(
        artifact_id=str(payload["artifact_id"]),
        source_corpora=tuple(payload["source_corpora"]),
        joint_names=names,
        joint_lower=lower,
        joint_upper=upper,
        mean=mean,
        components=components,
        coefficient_low=coefficient_low,
        coefficient_high=coefficient_high,
        cumulative_variance=cumulative_variance,
    )


def validate_host_joint_order(
    artifact: EigengraspArtifact,
    host_joint_names: Sequence[str],
) -> None:
    """Require the known left/right Sharpa name correspondence in exact order."""

    def remove_prefix(value: str, prefix: str) -> str:
        return value[len(prefix) :] if value.startswith(prefix) else value

    canonical_suffixes = tuple(
        remove_prefix(remove_prefix(name, "right_"), "left_")
        for name in artifact.joint_names
    )
    host_suffixes = tuple(
        remove_prefix(remove_prefix(name, "left_"), "right_")
        for name in host_joint_names
    )
    # Isaac Gym prefixes the five root joints to force deterministic URDF order.
    root_prefixes = {0: "1_", 5: "2_", 9: "3_", 13: "4_", 17: "5_"}
    host_suffixes = tuple(
        remove_prefix(suffix, root_prefixes.get(index, ""))
        for index, suffix in enumerate(host_suffixes)
    )
    if host_suffixes != canonical_suffixes:
        raise ValueError(
            "live Sharpa hand joint order differs from the eigengrasp artifact"
        )


def decode_absolute_eigengrasp(
    actions: torch.Tensor,
    *,
    mean: torch.Tensor,
    components: torch.Tensor,
    coefficient_low: torch.Tensor,
    coefficient_high: torch.Tensor,
    artifact_lower: torch.Tensor,
    artifact_upper: torch.Tensor,
    host_lower: torch.Tensor,
    host_upper: torch.Tensor,
    normalization: Normalization,
) -> torch.Tensor:
    """Decode bounded PCA coordinates into host-calibrated absolute targets."""

    if actions.ndim != 2:
        raise ValueError("eigengrasp actions must be a two-dimensional batch")
    k = actions.shape[1]
    if not 1 <= k <= components.shape[0]:
        raise ValueError("invalid eigengrasp action width")
    if normalization not in ("clamp", "gauge"):
        raise ValueError(f"unsupported eigengrasp normalization: {normalization}")
    if normalization == "gauge" and k != components.shape[0]:
        raise ValueError("gauge eigengrasp decoding requires the full-rank basis")

    action = torch.clamp(actions, -1.0, 1.0)
    low = coefficient_low[:k]
    high = coefficient_high[:k]
    coefficients = torch.where(action < 0.0, (-action) * low, action * high)
    offset = coefficients @ components[:k]

    if normalization == "gauge":
        radial_gauge = torch.amax(torch.abs(action), dim=-1, keepdim=True)
        box_ratios = torch.where(
            offset > 0.0,
            offset / (artifact_upper - mean),
            torch.where(
                offset < 0.0,
                offset / (artifact_lower - mean),
                torch.zeros_like(offset),
            ),
        )
        box_gauge = torch.amax(box_ratios, dim=-1, keepdim=True)
        offset = offset * (radial_gauge / torch.clamp_min(box_gauge, 1.0e-12))

    artifact_target = torch.clamp(mean + offset, artifact_lower, artifact_upper)
    normalized_target = (artifact_target - artifact_lower) / (
        artifact_upper - artifact_lower
    )
    host_target = host_lower + normalized_target * (host_upper - host_lower)
    return torch.clamp(host_target, host_lower, host_upper)


__all__ = [
    "EigengraspArtifact",
    "Normalization",
    "decode_absolute_eigengrasp",
    "load_eigengrasp_artifact",
    "validate_host_joint_order",
]
