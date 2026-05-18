"""Standalone prompt-learning utilities extracted from the CoOp trainer.

This module provides lightweight versions of the prompt learner components
(`PromptLearner` and `TextEncoder`) without importing the full trainer stack
that depends on external libraries such as Dassl.

The implementation mirrors the behaviour of `CoOp-main/trainers/prompt_learner.py`
so MMA can reuse the same prompt mechanics without mutating the original
repository or altering Python's module search path.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from CLIP.tokenizer import SimpleTokenizer as _Tokenizer, tokenize as clip_tokenize

__all__ = ["PromptLearner", "TextEncoder"]

_tokenizer = _Tokenizer()


def _resolve_clip_dtype(clip_model: nn.Module) -> torch.dtype:
    """Best-effort retrieval of the text tower's computation dtype."""

    if hasattr(clip_model, "dtype"):
        return clip_model.dtype

    transformer = getattr(clip_model, "transformer", None)
    if transformer is not None and hasattr(transformer, "get_cast_dtype"):
        return transformer.get_cast_dtype()

    text_projection = getattr(clip_model, "text_projection", None)
    if text_projection is not None:
        return text_projection.dtype

    first_param = next(clip_model.parameters(), None)
    if first_param is not None:
        return first_param.dtype

    return torch.float32


class TextEncoder(nn.Module):
    """Mirror CLIP's internal text encoder for prompt learning."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = getattr(clip_model, "attn_mask", None)
        self.dtype = _resolve_clip_dtype(clip_model)

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        try:
            x = self.transformer(x, attn_mask=self.attn_mask)
        except TypeError:
            x = self.transformer(x)

        if isinstance(x, (list, tuple)):
            x = x[0]  # keep the last hidden state

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest index)
        token_indices = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), token_indices]
        x = x @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """Learnable text prompt context as defined in CoOp."""

    def __init__(self, cfg, classnames: Sequence[str], clip_model: nn.Module, device=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = _resolve_clip_dtype(clip_model)
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if device is None:
            try:
                device = next(clip_model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        device = torch.device(device)

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip_tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt.to(device)).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(
                    n_cls,
                    n_ctx,
                    ctx_dim,
                    dtype=dtype,
                    device=device,
                )
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip_tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # SOS / CLS / EOS buffers mirror the trainer implementation
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self) -> torch.Tensor:
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1]
                class_i = suffix[i : i + 1, :name_len]
                suffix_i = suffix[i : i + 1, name_len:]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:]
                prompt = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1]
                class_i = suffix[i : i + 1, :name_len]
                suffix_i = suffix[i : i + 1, name_len:]
                ctx_i = ctx[i : i + 1]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError(
                f"Unsupported class token position: {self.class_token_position}"
            )

        return prompts
