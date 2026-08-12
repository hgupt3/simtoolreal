from pathlib import Path

import pytest
import torch

from isaacgymenvs.utils.eigengrasp import (
    decode_absolute_eigengrasp,
    load_eigengrasp_artifact,
    validate_host_joint_order,
)


ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "assets/eigengrasp/sharpa_arctic_30hz_v1_pca.json"
)
HOST_NAMES = (
    "left_1_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_2_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
)


def _decode(actions: torch.Tensor, normalization: str = "clamp") -> torch.Tensor:
    artifact = load_eigengrasp_artifact(ARTIFACT_PATH)
    # Match the intentionally restricted ab/adduction limits in SimToolReal.
    host_lower = artifact.joint_lower.clone()
    host_upper = artifact.joint_upper.clone()
    for index in (6, 10, 14, 19):
        host_lower[index] = -0.03491
        host_upper[index] = 0.03491
    return decode_absolute_eigengrasp(
        actions,
        mean=artifact.mean,
        components=artifact.components,
        coefficient_low=artifact.coefficient_low,
        coefficient_high=artifact.coefficient_high,
        artifact_lower=artifact.joint_lower,
        artifact_upper=artifact.joint_upper,
        host_lower=host_lower,
        host_upper=host_upper,
        normalization=normalization,
    )


def test_arctic_sharpa_artifact_contract() -> None:
    artifact = load_eigengrasp_artifact(ARTIFACT_PATH)
    assert artifact.artifact_id == "sharpa_arctic_30hz_v1_pca"
    assert artifact.components.shape == (22, 22)
    assert artifact.cumulative_variance[4].item() == pytest.approx(0.71365744)
    torch.testing.assert_close(
        artifact.components @ artifact.components.T,
        torch.eye(22),
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_live_left_sharpa_order_matches_canonical_right_artifact() -> None:
    artifact = load_eigengrasp_artifact(ARTIFACT_PATH)
    validate_host_joint_order(artifact, HOST_NAMES)
    swapped = list(HOST_NAMES)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    with pytest.raises(ValueError, match="joint order"):
        validate_host_joint_order(artifact, swapped)


@pytest.mark.parametrize("k", (5, 22))
def test_literal_decode_shape_zero_and_limits(k: int) -> None:
    artifact = load_eigengrasp_artifact(ARTIFACT_PATH)
    actions = torch.zeros(3, k)
    actions[1] = 1.0
    actions[2] = -1.0
    targets = _decode(actions)
    assert targets.shape == (3, 22)
    assert bool(torch.isfinite(targets).all())

    host_lower = artifact.joint_lower.clone()
    host_upper = artifact.joint_upper.clone()
    for index in (6, 10, 14, 19):
        host_lower[index] = -0.03491
        host_upper[index] = 0.03491
    expected_zero = host_lower + (
        (artifact.mean - artifact.joint_lower)
        / (artifact.joint_upper - artifact.joint_lower)
    ) * (host_upper - host_lower)
    torch.testing.assert_close(targets[0], expected_zero)
    assert bool(torch.all(targets >= host_lower))
    assert bool(torch.all(targets <= host_upper))


def test_gauge_full_rank_uses_requested_joint_box_radius() -> None:
    artifact = load_eigengrasp_artifact(ARTIFACT_PATH)
    actions = torch.randn(16, 22).clamp(-1.0, 1.0)
    actions[0].zero_()
    targets = _decode(actions, normalization="gauge")

    host_lower = artifact.joint_lower.clone()
    host_upper = artifact.joint_upper.clone()
    for index in (6, 10, 14, 19):
        host_lower[index] = -0.03491
        host_upper[index] = 0.03491
    host_mean = _decode(torch.zeros(1, 22))[0]
    offset = targets - host_mean
    ratios = torch.where(
        offset > 0.0,
        offset / (host_upper - host_mean),
        torch.where(
            offset < 0.0,
            offset / (host_lower - host_mean),
            torch.zeros_like(offset),
        ),
    )
    torch.testing.assert_close(
        ratios.amax(dim=-1), actions.abs().amax(dim=-1), atol=2.0e-5, rtol=2.0e-5
    )


def test_gauge_rejects_reduced_rank() -> None:
    with pytest.raises(ValueError, match="full-rank"):
        _decode(torch.zeros(2, 5), normalization="gauge")
