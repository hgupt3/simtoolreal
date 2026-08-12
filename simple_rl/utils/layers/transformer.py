from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for a causal, recurrent transformer policy.

    ``rolling`` gives every step a fixed sliding window. ``segment`` follows
    Transformer-XL: a complete, detached previous segment is combined with
    the causally masked current segment.
    """

    memory_mode: Literal["rolling", "segment"] = "rolling"
    block_type: Literal["qwen", "transformer_xl"] = "qwen"
    context_length: int = 16
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: Optional[int] = None
    ffn_hidden: Optional[int] = None
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    actor_head_units: Sequence[int] = ()
    critic_head_units: Sequence[int] = ()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [batch, heads, time, head_dim]; cos/sin: [time, head_dim / 2]
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.view(1, 1, cos.shape[0], cos.shape[1]).to(x.dtype)
    sin = sin.view(1, 1, sin.shape[0], sin.shape[1]).to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        qk_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("transformer embedding size must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                "transformer attention head dimension must be even for RoPE"
            )
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(
        self,
        query_input: torch.Tensor,
        key_value_input: torch.Tensor,
        attention_mask: torch.Tensor,
        query_rope: Tuple[torch.Tensor, torch.Tensor],
        key_rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, query_length, _ = query_input.shape
        key_length = key_value_input.shape[1]
        q = (
            self.q_proj(query_input)
            .view(batch_size, query_length, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key_value_input)
            .view(batch_size, key_length, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(key_value_input)
            .view(batch_size, key_length, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = _apply_rope(self.q_norm(q), *query_rope)
        k = _apply_rope(self.k_norm(k), *key_rope)
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(
            output.transpose(1, 2).reshape(batch_size, query_length, -1)
        )


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_size, bias=False)
        self.up = nn.Linear(d_model, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_size: int,
        dropout: float,
        block_type: Literal["qwen", "transformer_xl"],
    ) -> None:
        super().__init__()
        qwen = block_type == "qwen"
        norm = RMSNorm if qwen else nn.LayerNorm
        self.attention_norm = norm(d_model)
        self.attention = CausalAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            qk_norm=qwen,
            dropout=dropout,
        )
        self.ffn_norm = norm(d_model)
        self.ffn = (
            SwiGLU(d_model, hidden_size, dropout)
            if qwen
            else TransformerFeedForward(d_model, hidden_size, dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        query_rope: Tuple[torch.Tensor, torch.Tensor],
        key_rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        key_value = x if memory is None else torch.cat((memory, x), dim=1)
        normalized_key_value = self.attention_norm(key_value)
        normalized_query = normalized_key_value[:, -x.shape[1] :]
        x = x + self.attention(
            normalized_query,
            normalized_key_value,
            attention_mask,
            query_rope,
            key_rope,
        )
        return x + self.ffn(self.ffn_norm(x))


class StatefulTransformer(nn.Module):
    """Causal transformer with PPO-chunk-compatible detached memory state."""

    def __init__(self, config: TransformerConfig, d_model: int, num_seqs: int) -> None:
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.num_seqs = num_seqs
        self._validate()
        n_kv_heads = config.n_kv_heads or config.n_heads
        hidden_size = config.ffn_hidden or 2 * d_model
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                n_heads=config.n_heads,
                n_kv_heads=n_kv_heads,
                hidden_size=hidden_size,
                dropout=config.dropout,
                block_type=config.block_type,
            )
            for _ in range(config.n_layers)
        )
        self.final_norm = (
            RMSNorm(d_model) if config.block_type == "qwen" else nn.LayerNorm(d_model)
        )

        head_dim = d_model // config.n_heads
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        max_positions = 2 * config.context_length + 1
        frequencies = torch.outer(torch.arange(max_positions).float(), inv_freq)
        self.register_buffer("rope_cos", frequencies.cos(), persistent=False)
        self.register_buffer("rope_sin", frequencies.sin(), persistent=False)

        for module in self.blocks.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate(self) -> None:
        cfg = self.config
        if cfg.memory_mode not in ("rolling", "segment"):
            raise ValueError(f"unknown transformer memory_mode: {cfg.memory_mode}")
        if cfg.block_type not in ("qwen", "transformer_xl"):
            raise ValueError(f"unknown transformer block_type: {cfg.block_type}")
        if cfg.context_length <= 0 or cfg.n_layers <= 0 or cfg.n_heads <= 0:
            raise ValueError(
                "transformer context_length, n_layers, and n_heads must be positive"
            )
        if cfg.dropout != 0.0:
            raise ValueError(
                "transformer dropout must be 0 for rollout/training policy equivalence"
            )
        if cfg.n_kv_heads is not None and cfg.n_kv_heads <= 0:
            raise ValueError("transformer n_kv_heads must be positive")
        n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        if cfg.block_type == "transformer_xl" and n_kv_heads != cfg.n_heads:
            raise ValueError(
                "transformer_xl blocks use standard multi-head attention; n_kv_heads must equal n_heads"
            )
        if cfg.ffn_hidden is not None and cfg.ffn_hidden <= 0:
            raise ValueError("transformer ffn_hidden must be positive")
        if cfg.rope_theta <= 0:
            raise ValueError("transformer rope_theta must be positive")
        if any(unit <= 0 for unit in cfg.actor_head_units):
            raise ValueError("transformer actor_head_units must be positive")
        if any(unit <= 0 for unit in cfg.critic_head_units):
            raise ValueError("transformer critic_head_units must be positive")

    @property
    def memory_capacity(self) -> int:
        if self.config.memory_mode == "segment":
            return 2 * self.config.context_length - 1
        return self.config.context_length

    def get_default_state(self) -> Tuple[torch.Tensor, ...]:
        capacity = self.memory_capacity
        states: List[torch.Tensor] = [
            torch.zeros((capacity, self.num_seqs, self.d_model))
            for _ in range(self.config.n_layers)
        ]
        states.append(torch.zeros((capacity, self.num_seqs, 1)))
        if self.config.memory_mode == "segment":
            # Segment phase is global rollout structure, not episodic memory.
            states.append(torch.zeros((1, self.num_seqs, 1)))
        return tuple(states)

    def reset_state(
        self, states: Sequence[torch.Tensor], indices: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        indices = indices.reshape(-1)
        reset = list(states)
        # Preserve the final segment-phase tensor. Episode resets clear content,
        # not PPO's globally aligned segment boundary.
        content_states = reset[:-1] if self.config.memory_mode == "segment" else reset
        for state in content_states:
            state[:, indices, :] = 0
        return tuple(reset)

    def _rope(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rope_cos[start:end], self.rope_sin[start:end]

    def forward(
        self,
        embeddings: torch.Tensor,
        states: Sequence[torch.Tensor],
        is_train: bool,
        seq_length: int = 1,
        memory_resets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if is_train:
            if self.config.memory_mode == "segment":
                return self._train_segment(
                    embeddings, states, seq_length, memory_resets
                )
            return self._train_rolling(embeddings, states, seq_length, memory_resets)
        if self.config.memory_mode == "segment":
            return self._step_segment(embeddings, states)
        return self._step_rolling(embeddings, states)

    def _reset_segment_mask(
        self,
        memory_resets: Optional[torch.Tensor],
        batch_size: int,
        seq_length: int,
        memory_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        causal_current = torch.ones(
            seq_length, seq_length, dtype=torch.bool, device=device
        ).tril()
        if memory_resets is None:
            return torch.cat(
                (
                    torch.ones(
                        seq_length, memory_length, dtype=torch.bool, device=device
                    ),
                    causal_current,
                ),
                dim=1,
            ).view(1, 1, seq_length, memory_length + seq_length)
        resets = memory_resets.reshape(batch_size, seq_length).bool()
        segment = resets.long().cumsum(dim=1)
        memory_segment = torch.zeros(
            batch_size, memory_length, dtype=torch.long, device=device
        )
        all_segments = torch.cat((memory_segment, segment), dim=1)
        same_segment = segment.unsqueeze(2) == all_segments.unsqueeze(1)
        causal = torch.cat(
            (
                torch.ones(seq_length, memory_length, dtype=torch.bool, device=device),
                causal_current,
            ),
            dim=1,
        )
        return (same_segment & causal.unsqueeze(0)).unsqueeze(1)

    def _train_segment(
        self,
        embeddings: torch.Tensor,
        states: Sequence[torch.Tensor],
        seq_length: int,
        memory_resets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        context = self.config.context_length
        if seq_length != context:
            raise ValueError(
                f"segment transformer requires seq_length == context_length ({context}), got {seq_length}"
            )
        batch_size = embeddings.shape[0] // seq_length
        x = embeddings.reshape(batch_size, seq_length, self.d_model)
        memories, valid = states[: self.config.n_layers], states[self.config.n_layers]
        memories = [
            memory[-context:].to(x.dtype).transpose(0, 1) for memory in memories
        ]
        valid = valid[-context:].permute(1, 2, 0) > 0
        reset_mask = self._reset_segment_mask(
            memory_resets, batch_size, seq_length, context, x.device
        )
        key_valid = torch.cat(
            (
                valid,
                torch.ones(
                    batch_size, 1, seq_length, dtype=torch.bool, device=x.device
                ),
            ),
            dim=2,
        ).unsqueeze(2)
        attention_mask = reset_mask & key_valid
        query_rope = self._rope(context, context + seq_length)
        key_rope = self._rope(0, context + seq_length)
        for block, memory in zip(self.blocks, memories):
            x = block(x, memory, attention_mask, query_rope, key_rope)
        return self.final_norm(x).reshape(-1, self.d_model), tuple(states)

    def _train_rolling(
        self,
        embeddings: torch.Tensor,
        states: Sequence[torch.Tensor],
        seq_length: int,
        memory_resets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        context = self.config.context_length
        if seq_length > context:
            raise ValueError(
                "rolling transformer seq_length cannot exceed context_length"
            )
        batch_size = embeddings.shape[0] // seq_length
        x = embeddings.reshape(batch_size, seq_length, self.d_model)
        memories, valid = states[:-1], states[-1]
        total_length = context + seq_length
        key_index = torch.arange(total_length, device=x.device).view(1, -1)
        query_index = torch.arange(seq_length, device=x.device).view(-1, 1)
        is_memory = key_index < context
        # A rollout ring buffer evicts one old slot before every query.
        causal = (is_memory & (key_index >= query_index)) | (
            ~is_memory & (key_index - context <= query_index)
        )
        key_valid = torch.cat(
            (
                valid.permute(1, 2, 0) > 0,
                torch.ones(
                    batch_size, 1, seq_length, dtype=torch.bool, device=x.device
                ),
            ),
            dim=2,
        ).unsqueeze(2)
        reset_mask = self._reset_segment_mask(
            memory_resets, batch_size, seq_length, context, x.device
        )
        attention_mask = (
            causal.view(1, 1, seq_length, total_length) & key_valid & reset_mask
        )
        query_rope = self._rope(context, total_length)
        key_rope = self._rope(0, total_length)
        for block, memory in zip(self.blocks, memories):
            x = block(
                x,
                memory.to(x.dtype).transpose(0, 1),
                attention_mask,
                query_rope,
                key_rope,
            )
        return self.final_norm(x).reshape(-1, self.d_model), tuple(states)

    def _step_rolling(
        self, embeddings: torch.Tensor, states: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        memories, valid = states[:-1], states[-1]
        x = embeddings.to(memories[0].dtype).unsqueeze(1)
        key_valid = torch.cat(
            (
                valid.permute(1, 2, 0) > 0,
                torch.ones(x.shape[0], 1, 1, dtype=torch.bool, device=x.device),
            ),
            dim=2,
        ).unsqueeze(2)
        context = self.config.context_length
        query_rope = self._rope(context, context + 1)
        key_rope = self._rope(0, context + 1)
        new_states: List[torch.Tensor] = []
        for block, memory in zip(self.blocks, memories):
            new_states.append(torch.cat((memory[1:], x.transpose(0, 1)), dim=0))
            x = block(
                x, memory.to(x.dtype).transpose(0, 1), key_valid, query_rope, key_rope
            )
        new_states.append(torch.cat((valid[1:], torch.ones_like(valid[:1])), dim=0))
        return self.final_norm(x)[:, 0], tuple(new_states)

    def _step_segment(
        self, embeddings: torch.Tensor, states: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        memories = states[: self.config.n_layers]
        valid = states[self.config.n_layers]
        phase_state = states[-1]
        phases = phase_state[0, :, 0].round().long()
        if not torch.equal(phases, phases[:1].expand_as(phases)):
            raise ValueError(
                "segment phases must remain aligned across vector environments"
            )
        phase = int(phases[0].item())
        context = self.config.context_length
        memory_length = context + phase
        x = embeddings.to(memories[0].dtype).unsqueeze(1)
        selected_valid = valid[-memory_length:]
        key_valid = torch.cat(
            (
                selected_valid.permute(1, 2, 0) > 0,
                torch.ones(x.shape[0], 1, 1, dtype=torch.bool, device=x.device),
            ),
            dim=2,
        ).unsqueeze(2)
        query_rope = self._rope(memory_length, memory_length + 1)
        key_rope = self._rope(0, memory_length + 1)
        new_states: List[torch.Tensor] = []
        for block, memory in zip(self.blocks, memories):
            selected_memory = memory[-memory_length:].to(x.dtype).transpose(0, 1)
            new_states.append(torch.cat((memory[1:], x.transpose(0, 1)), dim=0))
            x = block(x, selected_memory, key_valid, query_rope, key_rope)
        new_states.append(torch.cat((valid[1:], torch.ones_like(valid[:1])), dim=0))
        next_phase = (phase + 1) % context
        new_states.append(torch.full_like(phase_state, float(next_phase)))
        return self.final_norm(x)[:, 0], tuple(new_states)
