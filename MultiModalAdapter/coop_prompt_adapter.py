"""Bridging helpers that let MMA reuse CoOp's text encoder without patching it."""

from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from MultiModalAdapter.clip.gsl import gradient_scale_layer


class CrossAttentionAdapter(nn.Module):
    """Lightweight cross-attention block with residual layer norm."""

    def __init__(self, query_dim: int, n_heads: int, key_dim: Optional[int] = None):
        super().__init__()

        kdim = key_dim if key_dim is not None else query_dim
        self.query_dim = query_dim
        self.key_dim = kdim

        self.mha = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=n_heads,
            batch_first=True,
            kdim=kdim,
            vdim=kdim,
        )
        self.ln = nn.LayerNorm(query_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Fuse ``query`` with ``key_value`` using cross-attention."""

        if query.dim() == 2:
            query = query.unsqueeze(0)
        elif query.dim() != 3:
            raise ValueError("query must have shape [N, D] or [B, N, D]")

        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(0)
        elif key_value.dim() != 3:
            raise ValueError("key_value must have shape [M, D] or [B, M, D]")

        if key_value.size(0) == 1 and query.size(0) > 1:
            key_value = key_value.expand(query.size(0), -1, -1)

        key_value = key_value.to(query.dtype)

        attn_output, _ = self.mha(query, key_value, key_value)
        fused_query = self.ln(query + attn_output)
        return fused_query.squeeze(0) if fused_query.size(0) == 1 else fused_query


class CrossAttentionBundle(nn.Module):
    """Container that stores cross-attention adapters for joint layers."""

    def __init__(
        self,
        text_dim: int,
        n_heads: int,
        joint_layers: Iterable[int],
        vision_dim: Optional[int] = None,
    ):
        super().__init__()
        keys = [str(layer) for layer in sorted(set(joint_layers))]
        self.register_buffer(
            "_layer_index", torch.tensor(sorted(set(joint_layers))), persistent=False
        )
        vision_dim = vision_dim if vision_dim is not None else text_dim
        self.v_to_t = nn.ModuleDict(
            {key: CrossAttentionAdapter(text_dim, n_heads, key_dim=vision_dim) for key in keys}
        )
        self.t_to_v = nn.ModuleDict(
            {key: CrossAttentionAdapter(vision_dim, n_heads, key_dim=text_dim) for key in keys}
        )

    def has_layer(self, layer_idx: int) -> bool:
        return str(layer_idx) in self.v_to_t and str(layer_idx) in self.t_to_v

    def get_v_to_t(self, layer_idx: int) -> Optional[CrossAttentionAdapter]:
        key = str(layer_idx)
        return self.v_to_t[key] if key in self.v_to_t else None

    def get_t_to_v(self, layer_idx: int) -> Optional[CrossAttentionAdapter]:
        key = str(layer_idx)
        return self.t_to_v[key] if key in self.t_to_v else None


def _apply_text_adapter_residual(
    block: nn.Module,
    hidden: torch.Tensor,
    adapter_func: Optional[Callable],
    layer_idx: int,
) -> torch.Tensor:
    if adapter_func is None:
        return hidden

    seq_adapter, shared_adapter, scale = adapter_func(layer_idx)
    if seq_adapter is None:
        return hidden

    dtype = hidden.dtype
    residual = gradient_scale_layer(hidden, scale)
    residual = block.ln_1(residual)
    residual = seq_adapter.down(residual)
    if shared_adapter is not None:
        residual = shared_adapter(residual)
    residual = seq_adapter.up(residual) * scale
    residual = gradient_scale_layer(residual, 1.0 / scale)
    residual = residual.to(dtype)
    return hidden + residual


def _forward_block(
    block: nn.Module, hidden: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    output = block(hidden, attn_mask=attn_mask) if attn_mask is not None else block(hidden)
    if isinstance(output, tuple):
        return output[0]
    return output


def encode_prompts_with_optional_adapter(
    text_encoder,
    prompts: torch.Tensor,
    tokenized_prompts: torch.Tensor,
    adapter_func: Optional[Callable] = None,
) -> torch.Tensor:
    """Run CoOp's TextEncoder with an MMA adapter function if provided.

    The helper mirrors :meth:`TextEncoder.forward` but injects MMA's residual
    adapters by calling the underlying CLIP transformer with the
    ``adapter_func`` signature expected by MMA.  This keeps the original CoOp
    module untouched while still letting MMA share the same encoder instance.
    """

    dtype = getattr(text_encoder, "dtype", prompts.dtype)
    positional_embedding = text_encoder.positional_embedding.type(dtype)

    x = prompts + positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND

    attn_mask = getattr(text_encoder, "attn_mask", None)
    if attn_mask is None:
        attn_mask = getattr(text_encoder.transformer, "attn_mask", None)
    for layer_idx, block in enumerate(text_encoder.transformer.resblocks, start=1):
        x = _forward_block(block, x, attn_mask)
        x = _apply_text_adapter_residual(block, x, adapter_func, layer_idx)

    x = x.permute(1, 0, 2)  # LND -> NLD
    x = text_encoder.ln_final(x).type(dtype)

    x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
    x = x @ text_encoder.text_projection
    return x


def encode_prompts_with_layer_context(
    text_encoder,
    prompts: torch.Tensor,
    tokenized_prompts: torch.Tensor,
    adapter_func: Optional[Callable] = None,
    layer_bias: Optional[Mapping[int, torch.Tensor]] = None,
    hook_layers: Optional[Iterable[int]] = None,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Run the shared text encoder while collecting hidden states for select layers.

    Parameters
    ----------
    text_encoder:
        The CoOp text encoder instance.
    prompts:
        Prompt tensor emitted by ``PromptLearner``.
    tokenized_prompts:
        Token indices that identify the ``eot`` position for projection.
    adapter_func:
        Optional MMA adapter callback. When provided the transformer blocks
        expect the ``[activations, adapter_func]`` pair.
    layer_bias:
        Mapping from transformer layer index (1-based) to a tensor that should
        be added to the post-layer activations. This allows the vision branch to
        inject high-level context before the text pipeline continues.
    hook_layers:
        Iterable of layer indices whose activations should be returned for
        downstream fusion. When ``None`` all layers specified in ``layer_bias``
        are collected.

    Returns
    -------
    text_features:
        Normalised CLIP text features.
    layer_states:
        Dictionary mapping layer index to the activations immediately after the
        residual block. Shapes follow ``[n_prompts, n_ctx, hidden_dim]`` so the
        caller can derive summary statistics for MMA fusion.
    """

    dtype = getattr(text_encoder, "dtype", prompts.dtype)
    positional_embedding = text_encoder.positional_embedding.type(dtype)

    x = prompts + positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND

    payload = x

    requested_layers = set(hook_layers or [])
    if not requested_layers and layer_bias is not None:
        requested_layers = set(layer_bias.keys())

    collected: Dict[int, torch.Tensor] = {}
    bias_mapping = layer_bias or {}

    attn_mask = getattr(text_encoder, "attn_mask", None)
    if attn_mask is None:
        attn_mask = getattr(text_encoder.transformer, "attn_mask", None)

    for layer_idx, block in enumerate(text_encoder.transformer.resblocks, start=1):
        payload = _forward_block(block, payload, attn_mask)
        payload = _apply_text_adapter_residual(block, payload, adapter_func, layer_idx)

        hidden = payload

        bias = bias_mapping.get(layer_idx)
        if bias is not None:
            bias_tensor = bias.to(hidden.dtype)
            while bias_tensor.dim() < hidden.dim():
                bias_tensor = bias_tensor.unsqueeze(0)
            hidden = hidden + bias_tensor

        if layer_idx in requested_layers:
            collected[layer_idx] = hidden.permute(1, 0, 2)

    x = payload

    x = x.permute(1, 0, 2)  # LND -> NLD
    x = text_encoder.ln_final(x).type(dtype)
    x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
    x = x @ text_encoder.text_projection

    return x, collected


def encode_prompts_with_cross_attention(
    text_encoder,
    prompts: torch.Tensor,
    tokenized_prompts: torch.Tensor,
    adapter_func: Optional[Callable] = None,
    patch_token_map: Optional[Mapping[int, torch.Tensor]] = None,
    cross_attention: Optional[CrossAttentionBundle] = None,
    joint_layers: Optional[Iterable[int]] = None,
    mma_text_fusion_scale: float = 0.1,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Encode prompts while exchanging context with vision tokens via cross-attention."""

    dtype = getattr(text_encoder, "dtype", prompts.dtype)
    positional_embedding = text_encoder.positional_embedding.type(dtype)

    x = prompts + positional_embedding
    x = x.permute(1, 0, 2)

    payload = x

    patch_token_map = patch_token_map or {}
    joint_layer_set = set(joint_layers or [])
    fused_states: Dict[int, torch.Tensor] = {}

    attn_mask = getattr(text_encoder, "attn_mask", None)
    if attn_mask is None:
        attn_mask = getattr(text_encoder.transformer, "attn_mask", None)

    for layer_idx, block in enumerate(text_encoder.transformer.resblocks, start=1):
        payload = _forward_block(block, payload, attn_mask)
        payload = _apply_text_adapter_residual(block, payload, adapter_func, layer_idx)
        hidden = payload

        if layer_idx in joint_layer_set:
            text_tokens_original = hidden.permute(1, 0, 2)
            patch_tokens = patch_token_map.get(layer_idx)
            text_tokens_fused = text_tokens_original
            if patch_tokens is not None and cross_attention is not None:
                adapter = cross_attention.get_v_to_t(layer_idx)
                if adapter is not None:
                    adapter_output = adapter(text_tokens_original, patch_tokens)
                    text_tokens_fused = text_tokens_original + adapter_output * mma_text_fusion_scale
            fused_states[layer_idx] = text_tokens_fused
            hidden = text_tokens_fused.permute(1, 0, 2)
            payload = hidden

    final_hidden = payload
    final_hidden = final_hidden.permute(1, 0, 2)
    final_hidden = text_encoder.ln_final(final_hidden).type(dtype)
    final_hidden = final_hidden[
        torch.arange(final_hidden.shape[0]), tokenized_prompts.argmax(dim=-1)
    ]
    final_hidden = final_hidden @ text_encoder.text_projection

    return final_hidden, fused_states
