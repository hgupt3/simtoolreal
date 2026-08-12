"""Plot reward curves from locally downloaded TensorBoard event directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = {
    "rlgames-lstm-sapg-seed0": ("rl_games LSTM", "#4d4d4d"),
    "legacy-simplerl-lstm-sapg-seed0": ("Legacy simple_rl LSTM", "#e68613"),
    "current-simplerl-lstm-sapg-seed0": ("Current simple_rl LSTM", "#3478bf"),
    "current-simplerl-rolling16-sapg-seed0": ("Rolling Transformer (16)", "#1b9e77"),
    "current-simplerl-loco128-sapg-seed0": ("LocoFormer / TXL (128)", "#b04cc2"),
}


def load_scalars(run_dir: Path, tag: str = "rewards/step") -> tuple[np.ndarray, np.ndarray]:
    """Merge restarted event files, keeping the newest value at duplicate steps."""
    newest_by_step = {}
    for event_file in run_dir.glob("events.out.tfevents.*"):
        accumulator = EventAccumulator(
            str(event_file), size_guidance={"scalars": 0}
        ).Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            previous = newest_by_step.get(event.step)
            if previous is None or event.wall_time >= previous.wall_time:
                newest_by_step[event.step] = event
    events = [newest_by_step[step] for step in sorted(newest_by_step)]
    if not events:
        raise RuntimeError(f"No {tag!r} events found in {run_dir}")
    return (
        np.asarray([event.step for event in events], dtype=np.int64),
        np.asarray([event.value for event in events], dtype=np.float64),
    )


def frame_window_mean(
    frames: np.ndarray, values: np.ndarray, window_frames: int
) -> np.ndarray:
    """Trailing mean over a fixed number of frames, independent of log cadence."""
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    starts = np.searchsorted(frames, frames - window_frames, side="left")
    ends = np.arange(1, len(values) + 1)
    return (prefix[ends] - prefix[starts]) / (ends - starts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window-frames", type=int, default=25_000_000)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    curves = {}
    for run, (label, color) in RUNS.items():
        frames, rewards = load_scalars(args.event_root / run)
        curves[run] = (label, color, frames, rewards)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for label, color, frames, rewards in curves.values():
        x = frames / 1e9
        smoothed = frame_window_mean(frames, rewards, args.window_frames)
        ax.plot(x, rewards, color=color, alpha=0.07, linewidth=0.55)
        ax.plot(x, smoothed, color=color, linewidth=2.4, label=label)
        ax.scatter(x[-1], smoothed[-1], color=color, s=24, zorder=3)

    common_frames = min(curve[2][-1] for curve in curves.values())
    ax.axvline(common_frames / 1e9, color="#777777", linestyle="--", linewidth=1)
    ax.text(
        common_frames / 1e9,
        0.02,
        " common observed range",
        transform=ax.get_xaxis_transform(),
        color="#666666",
        fontsize=9,
        rotation=90,
        va="bottom",
        ha="right",
    )
    ax.set_title("SimToolReal SAPG reward curves — seed 0")
    ax.set_xlabel("Environment frames (billions)")
    ax.set_ylabel("Episode reward")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.text(
        0.01,
        0.99,
        f"Thick lines: trailing {args.window_frames / 1e6:g}M-frame mean; faint lines: raw logged reward",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))

    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "label", "frame", "reward", "smoothed_reward"])
        for run, (label, _, frames, rewards) in curves.items():
            smoothed = frame_window_mean(frames, rewards, args.window_frames)
            writer.writerows(
                (run, label, int(frame), float(reward), float(smooth))
                for frame, reward, smooth in zip(frames, rewards, smoothed)
            )


if __name__ == "__main__":
    main()
