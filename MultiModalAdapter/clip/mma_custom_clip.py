"""Minimal MMA wrapper that mirrors ``trainers.mmadapter.CustomCLIP`` without Dassl."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import clip
from CLIP.tokenizer import tokenize as clip_tokenize

__all__ = ["LayerAdapterDict", "AdapterLearner", "CustomCLIP", "TextEncoder"]


class LayerAdapterDict(nn.Module):
    """Container that registers per-layer adapters and tolerates missing indices."""

    def __init__(self):
        super().__init__()
        self._registry = nn.ModuleDict()

    def add(self, index: int, module: nn.Module) -> None:
        self._registry[str(index)] = module

    def get(self, index: int) -> Optional[nn.Module]:
        key = str(index)
        if key in self._registry:
            return self._registry[key]
        return None

    def items(self):
        for key, module in self._registry.items():
            yield int(key), module

    def __contains__(self, index: int) -> bool:
        return str(index) in self._registry

    def __len__(self) -> int:
        return len(self._registry)


def _infer_text_dtype(clip_model: nn.Module) -> torch.dtype:
    if hasattr(clip_model, "dtype"):
        return clip_model.dtype

    candidates = [
        getattr(getattr(clip_model, "ln_final", None), "weight", None),
        getattr(getattr(clip_model, "token_embedding", None), "weight", None),
    ]
    for tensor in candidates:
        if tensor is not None:
            return tensor.dtype

    try:
        first_param = next(clip_model.parameters())
        return first_param.dtype
    except StopIteration:
        return torch.float32


def _infer_visual_dtype(clip_model: nn.Module) -> torch.dtype:
    visual = getattr(clip_model, "visual", None)
    if visual is None:
        return _infer_text_dtype(clip_model)

    candidates = []
    for attr in ("proj", "ln_post", "conv1"):
        module = getattr(visual, attr, None)
        if module is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is not None:
            candidates.append(weight)
    if hasattr(visual, "parameters"):
        try:
            candidates.append(next(visual.parameters()))
        except StopIteration:
            pass

    for tensor in candidates:
        if tensor is not None:
            return tensor.dtype

    return torch.float32


class TextEncoder(nn.Module):
    """Lightweight wrapper around CLIP's text transformer."""

    def __init__(self, clip_model, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = dtype or _infer_text_dtype(clip_model)

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor, adapter_func=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        if adapter_func is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, adapter_func])
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        indices = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), indices] @ self.text_projection
        return x


class BottleneckAdapterBlock(nn.Module):
    """Adapter block with explicit ``down``/``up`` paths regardless of width."""

    def __init__(self, d_model: int, mid_dim: int, dtype: torch.dtype):
        super().__init__()

        self.down = nn.Sequential(nn.Linear(d_model, mid_dim), nn.ReLU())
        self.up = nn.Identity() if mid_dim == d_model else nn.Linear(mid_dim, d_model)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(module.bias, 0)

        if dtype == torch.float16:
            self.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.up(x)


def _build_adapter_block(d_model: int, mid_dim: int, dtype: torch.dtype) -> nn.Module:
    return BottleneckAdapterBlock(d_model, mid_dim, dtype)


class AdapterLearner(nn.Module):
    """Reproduces MMA's prompt/adapter stacks without Dassl dependencies."""

    def __init__(self, cfg, classnames: Sequence[str], clip_model):
        super().__init__()

        mm_cfg = cfg.TRAINER.MMADAPTER
        self.adapter_scale = float(getattr(mm_cfg, "ADAPTER_SCALE", 1.0))

        self.text_adapter = LayerAdapterDict()
        self.visual_adapter = LayerAdapterDict()
        self.shared_adapter = LayerAdapterDict()

        # 必须给 JOINT_LAYERS
        joint_layers = getattr(mm_cfg, "JOINT_LAYERS", None)
        if not joint_layers:
            raise ValueError("MMA requires JOINT_LAYERS when using sparse adapter placement.")
        target_layers = sorted(set(int(l) for l in joint_layers))

        # 维度推断
        adapter_dim = int(getattr(mm_cfg, "ADAPTER_DIM", clip_model.ln_final.weight.shape[0]))

        visual_width_override = getattr(mm_cfg, "VISUAL_WIDTH", None)
        if visual_width_override is not None:
            visual_width = int(visual_width_override)
        else:
            visual_module = getattr(clip_model, "visual", None)
            ln_post = getattr(visual_module, "ln_post", None) if visual_module is not None else None
            if ln_post is not None and getattr(ln_post, "weight", None) is not None:
                visual_width = ln_post.weight.shape[0]
            elif visual_module is not None and hasattr(visual_module, "width"):
                visual_width = int(visual_module.width)
            else:
                visual_width = adapter_dim

        text_dtype = _infer_text_dtype(clip_model)
        visual_dtype = _infer_visual_dtype(clip_model)

        # 只在 joint 层建 adapter
        for layer_idx in target_layers:
            text_block = _build_adapter_block(
                clip_model.ln_final.weight.shape[0], adapter_dim, text_dtype
            )
            visual_block = _build_adapter_block(
                visual_width, adapter_dim, visual_dtype
            )
            shared_block = _build_adapter_block(adapter_dim, adapter_dim, text_dtype)

            self.text_adapter.add(layer_idx, text_block)
            self.visual_adapter.add(layer_idx, visual_block)
            self.shared_adapter.add(layer_idx, shared_block)

        # 构造 CoOp 的 prompt embedding
        text_ctx_init = getattr(mm_cfg, "TEXT_CTX_INIT", "")
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [text_ctx_init + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip_tokenize(p) for p in prompts])
        token_device = clip_model.token_embedding.weight.device
        tokenized_prompts = tokenized_prompts.to(token_device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).to(text_dtype)

        self.register_buffer("token_embedding", embedding)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        self.text_adapter_func = lambda index: self.return_text_adapter(index)
        self.visual_adapter_func = lambda index: self.return_visual_adapter(index)

    def _lookup(self, container: LayerAdapterDict, index: int) -> Optional[nn.Module]:
        return container.get(index)

    def return_text_adapter(self, index: int) -> Tuple[Optional[nn.Module], Optional[nn.Module], float]:
        return self._lookup(self.text_adapter, index), self._lookup(self.shared_adapter, index), self.adapter_scale

    def return_visual_adapter(self, index: int) -> Tuple[Optional[nn.Module], Optional[nn.Module], float]:
        return self._lookup(self.visual_adapter, index), self._lookup(self.shared_adapter, index), self.adapter_scale

    def forward(self):
        return self.token_embedding, self.text_adapter_func, self.visual_adapter_func


class CustomCLIP(nn.Module):
    """Lightweight CLIP wrapper exposing adapter learners without Dassl."""

    def __init__(self, cfg, classnames: Sequence[str], clip_model):
        super().__init__()
        self.text_dtype = _infer_text_dtype(clip_model)
        self.visual_dtype = _infer_visual_dtype(clip_model)

        self.adapter_learner = AdapterLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.adapter_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, dtype=self.text_dtype)
        self.logit_scale = clip_model.logit_scale
        self.dtype = self.text_dtype
        self.text_features_for_inference = None

    def encode_text(self, prompts, tokenized_prompts, text_adapter_func=None):
        return self.text_encoder(prompts, tokenized_prompts, text_adapter_func)

    def encode_image(self, image, visual_adapter_func=None):
        image = image.to(self.visual_dtype)
        if visual_adapter_func is not None:
            return self.image_encoder([image, visual_adapter_func])
        return self.image_encoder(image)

    def forward(self, image):
        token_embedding, text_adapter_func, visual_adapter_func = self.adapter_learner()
        tokenized_prompts = self.tokenized_prompts
        if self.adapter_learner.training:
            text_features = self.encode_text(token_embedding, tokenized_prompts, text_adapter_func)
        else:
            if self.text_features_for_inference is None:
                self.text_features_for_inference = self.encode_text(
                    token_embedding, tokenized_prompts, text_adapter_func
                )
            text_features = self.text_features_for_inference
        image_features = self.encode_image(image, visual_adapter_func)
        return text_features, image_features, self.logit_scale.exp()
