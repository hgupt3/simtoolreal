from typing import Any, Optional, Tuple, Union

import torch
from torch import nn


def multiply_hidden(
    h: Union[torch.Tensor, Tuple[Any, ...]], mask: torch.Tensor
) -> Union[torch.Tensor, Tuple[Any, ...]]:
    if isinstance(h, torch.Tensor):
        return h * mask
    else:
        return tuple(multiply_hidden(v, mask) for v in h)


class RnnWithDones(nn.Module):
    def __init__(self, rnn_layer: nn.Module) -> None:
        nn.Module.__init__(self)
        self.rnn = rnn_layer

    # got idea from ikostrikov :)
    def forward(
        self,
        input: torch.Tensor,
        states: Any,
        done_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any]:
        if done_masks is None:
            return self.rnn(input, states)

        max_steps = input.size()[0]
        not_dones = ~done_masks.bool()
        # Preserve both time and batch axes even when either has length one.
        if max_steps > 1:
            reset_times = (~not_dones[1:].reshape(max_steps - 1, -1)).any(dim=1)
            has_zeros = (
                (reset_times.nonzero(as_tuple=False).flatten() + 1).cpu().tolist()
            )
        else:
            has_zeros = []

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [max_steps]
        out_batch = []

        for i in range(len(has_zeros) - 1):
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]
            not_done = not_dones[start_idx].to(input.dtype).reshape(1, -1, 1)
            states = multiply_hidden(states, not_done)
            out, states = self.rnn(input[start_idx:end_idx], states)
            out_batch.append(out)
        return torch.cat(out_batch, dim=0), states
