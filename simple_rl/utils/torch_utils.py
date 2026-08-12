import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float64"): torch.float32,
    np.dtype("float32"): torch.float32,
    # np.dtype('float64')    : torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}

torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    return action * d + m


def policy_kl(
    p0_mu: torch.Tensor,
    p0_sigma: torch.Tensor,
    p1_mu: torch.Tensor,
    p1_sigma: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    if reduce:
        return kl.mean()
    else:
        return kl


def safe_filesystem_op(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    This is to prevent spurious crashes related to saving checkpoints or restoring from checkpoints in a Network
    Filesystem environment (i.e. NGC cloud or SLURM)
    """
    num_attempts = 5
    for attempt in range(num_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(
                f"Exception {exc} when trying to execute {func} with args:{args} and kwargs:{kwargs}..."
            )
            wait_sec = 2**attempt
            print(f"Waiting {wait_sec} before trying again...")
            time.sleep(wait_sec)

    raise RuntimeError(
        f"Could not execute {func}, give up after {num_attempts} attempts..."
    )


def safe_save(state: Any, filename: Path) -> Any:
    return safe_filesystem_op(torch.save, state, filename)


def safe_load(
    filename: Path, map_location: Optional[Union[str, torch.device]] = "cpu"
) -> Any:
    from functools import partial

    load_fn = partial(torch.load, map_location=map_location)
    return safe_filesystem_op(load_fn, filename)


def save_checkpoint(filename: Path, state: Any) -> None:
    print(f"=> saving checkpoint '{filename}'")
    safe_save(state, filename)


def load_checkpoint(
    filename: Path, map_location: Optional[Union[str, torch.device]] = "cpu"
) -> Any:
    print(f"=> loading checkpoint '{filename}'")
    state = safe_load(filename, map_location=map_location)
    return state


def distributed_mean_var_count(
    mean: torch.Tensor, var: torch.Tensor, count: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    square_sum = var * (count - 1) + (mean**2) * count
    sum = mean * count
    count = torch.tensor(count, dtype=torch.float64).to(mean.device)
    dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)

    new_count = int(count.item())
    new_mean = sum / new_count
    new_var = (square_sum - (new_mean**2) * new_count) / (new_count - 1)
    return new_mean, new_var, new_count


class AverageMeter(nn.Module):
    def __init__(self, in_shape: Union[int, Tuple[int, ...]], max_size: int) -> None:
        """Initialize AverageMeter to track running average of values.

        This class maintains a running average of the most recent values up to max_size.
        The mean is tracked as a tensor of shape in_shape where in_shape is the shape of each value.
        When update() is called with values of shape (M, *in_shape), it updates the running average
        considering at most max_size previous values.

        Args:
            in_shape: Shape of each value (D,)
            max_size: Maximum number of previous values to consider
        """
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.register_buffer("current_size", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values: torch.Tensor) -> None:
        assert values.shape[1:] == self.mean.shape, (
            f"in_shape: {self.mean.shape}, values.shape: {values.shape}"
        )
        M = values.shape[0]
        if M == 0:
            return

        new_mean = torch.mean(values.float(), dim=0)
        M = min(M, self.max_size)
        old_size = min(int(self.current_size.item()), self.max_size - M)
        new_size = old_size + M

        self.current_size.fill_(new_size)
        self.mean.copy_((self.mean * old_size + new_mean * M) / new_size)

    def clear(self) -> None:
        self.current_size.fill_(0)
        self.mean.fill_(0)

    def __len__(self) -> int:
        return self.current_size.item()

    def get_mean(self) -> np.ndarray:
        return self.mean.cpu().numpy()
