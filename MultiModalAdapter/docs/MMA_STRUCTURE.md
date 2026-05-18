# Multi-Modal Adapter (MMA) Module Overview

This document summarizes the structure of the `MultiModalAdapter` package and highlights
integration points that differ from the existing MVFA (vision adapter with residual
connections) and CoOp (learnable textual prompts) baselines. It is intended to guide the
clean integration of MMA alongside MVFA and CoOp within a shared codebase.

## High-level folder layout

- `train.py` – CLI entry point that extends Dassl's default configuration with MMA-specific
  options and registers the custom trainer. It imports all dataset definitions so that they
  can be resolved at runtime. 【F:MultiModalAdapter/train.py†L1-L146】
- `configs/trainers/MultiModalAdapter/*.yaml` – Reference configurations for ViT-Backbone
  training, including MMA adapter hyper-parameters (layer range, bottleneck width, scaling,
  and precision mode). 【F:MultiModalAdapter/configs/trainers/MultiModalAdapter/vit_b16_ep5.yaml†L1-L38】
- `clip/mma_custom_clip.py` – Lightweight copy of the MMA `CustomCLIP` wrapper and
  `AdapterLearner` extracted for projects that do not depend on Dassl. It mirrors the adapter
  stacks and prompt embedding utilities while leaving training loops to the caller.
  【F:MultiModalAdapter/clip/mma_custom_clip.py†L1-L139】
- `trainers/mmadapter.py` – Original Dassl-based trainer and model wrapper. It downloads a
  frozen CLIP backbone, builds the adapter learner, and owns the optimization loop.
  【F:MultiModalAdapter/trainers/mmadapter.py†L14-L329】
- `clip/` – Fork of the CLIP model that exposes hooks for MMA adapters. It augments both the
  transformer blocks and the vision transformer stem to accept adapter functions and includes
  a gradient-scaling helper. 【F:MultiModalAdapter/clip/model.py†L168-L399】【F:MultiModalAdapter/clip/gsl.py†L1-L16】
- `datasets/`, `scripts/`, `lpclip/`, etc. – Reused from CoOp; not MMA-specific but required
  by the trainer entry point.

## Training entry and configuration flow

1. `train.py` calls `extend_cfg` to append `TRAINER.MMADAPTER` settings (prompt prefix,
   adapter start/end layers, bottleneck dimension, and gradient scale). 【F:MultiModalAdapter/train.py†L88-L97】
2. After command-line and YAML overrides are merged, Dassl builds the `MultiModalAdapter`
   trainer registered in `trainers/mmadapter.py`. Projects that bypass Dassl can instead import
   the lightweight wrapper in `clip/mma_custom_clip.py`. 【F:MultiModalAdapter/train.py†L100-L145】【F:MultiModalAdapter/clip/mma_custom_clip.py†L1-L139】
3. During `build_model`, the trainer downloads a CLIP checkpoint, upgrades it to the MMA-aware
   model, freezes the original CLIP parameters, and only optimizes the adapter learner. 【F:MultiModalAdapter/trainers/mmadapter.py†L214-L256】
4. Mixed precision and learning-rate scheduling are delegated to Dassl utilities; no CLIP
   internals are modified at runtime beyond the injected adapters. 【F:MultiModalAdapter/trainers/mmadapter.py†L256-L287】

## Adapter learner design

The MMA module wraps CLIP through three cooperating components defined in
`clip/mma_custom_clip.py` (mirroring the implementations inside `trainers/mmadapter.py`):

- **TextEncoder** – Shares CLIP's transformer but optionally accepts an adapter callback.
  When the callback is present the integration layer now replays MMA's residual bottleneck
  math after each transformer block instead of forwarding `[activations, adapter_func]`
  through the module itself. This keeps the public CLIP APIs untouched while yielding the
  same gradient-scaled residuals defined by MMA. The implementation mirrors the lightweight
  CoOp text encoder so the MVFA + CoOp pipeline can reference the exact same CLIP text tower
  (and its learnable token contexts) when MMA is introduced. The shim now reuses
  `CLIP.tokenizer.SimpleTokenizer` together with `CLIP.tokenizer.tokenize`, avoiding optional
  subpackages while matching the original CoOp behaviour.【F:MultiModalAdapter/clip/mma_custom_clip.py†L41-L108】【F:MultiModalAdapter/coop_prompt_learner.py†L1-L130】
- **AdapterLearner** – Replaces CoOp's prompt learner. It tokenizes class names with a
  configurable textual prefix, constructs parallel adapter stacks for text, vision, and a
  shared bridge, and exposes lightweight callbacks (`return_text_adapter` /
  `return_visual_adapter`) that retrieve per-layer adapter modules and the global scaling
  factor.【F:MultiModalAdapter/clip/mma_custom_clip.py†L135-L193】 The learner now stages the
  tokenized prompts on the CLIP token-embedding device before sampling embeddings, preventing
  CPU/GPU mismatches when the backbone runs on CUDA.【F:MultiModalAdapter/clip/mma_custom_clip.py†L176-L190】
  - `_build_adapter_block` instantiates per-layer bottlenecks only for the configured range,
    using `_infer_text_dtype` / `_infer_visual_dtype` to mirror CLIP's precision and keep the
    adapters aligned with the backbone for both fp16 and fp32 execution.【F:MultiModalAdapter/clip/mma_custom_clip.py†L41-L167】
  - The forward pass keeps layer index `0` accessible so MMA can perturb the textual context
    vector before it enters the transformer, something CoOp does not do. 【F:MultiModalAdapter/clip/mma_custom_clip.py†L126-L139】
- **CustomCLIP** – Houses the frozen CLIP encoders and routes images/text through the adapter
  learner. It memoizes text features for eval mode and always normalizes logits the same way as
  vanilla CLIP so downstream code remains unchanged.【F:MultiModalAdapter/clip/mma_custom_clip.py†L196-L233】

This structure generalizes CoOp: instead of optimizing a small set of learnable prompt
vectors, MMA optimizes module lists that sit both before and inside the transformers,
mirroring MVFA's philosophy of residual adapters but applied to both modalities with a shared
bridge.

> **Text encoder compatibility.** Because both the MMA `TextEncoder` and CoOp's
> `TextEncoder` directly wrap the same CLIP transformer, you should reuse the
> existing CoOp text encoder object (or at minimum pass the identical `clip_model`
> instance) when wiring MMA into the MVFA + CoOp training script. This preserves
> the learnable prompt tokens already injected by CoOp while letting MMA append
> adapters around the shared transformer stack.【F:MultiModalAdapter/clip/mma_custom_clip.py†L86-L108】【F:MultiModalAdapter/coop_prompt_learner.py†L33-L134】
> The integration layer accomplishes this with the
> `encode_prompts_with_optional_adapter` helper, which applies MMA's adapter
> bottlenecks after each CLIP transformer block so CoOp's Python module stays
> untouched while still receiving the same residual updates the Dassl trainer
> would provide.【F:MultiModalAdapter/coop_prompt_adapter.py†L1-L126】
>
> **Prompt ownership.** The learnable tokens and downstream text features must
> continue to originate from CoOp. MMA adapters add residual signals around the
> shared transformer but never replace CoOp's prompt generation or text encoding
> responsibilities. Maintaining this separation keeps training dynamics aligned
> with existing CoOp checkpoints and avoids vocabulary drift during evaluation.

## Modifications to CLIP internals

MMA relies on surgical changes in `clip/model.py` to accept adapter functions without
rewriting CLIP's main code path:

- `ModifiedResidualAttentionBlock.forward` detects when it receives `[x, adapter_func]`,
  fetches the layer-specific sequential adapter plus the shared module, applies gradient
  scaling before and after the bottleneck, and re-injects the adapted residual back into the
  transformer block. 【F:MultiModalAdapter/clip/model.py†L188-L215】
- `VisionTransformer.forward` mirrors this logic at layer index 0, modifying the patch
  embeddings prior to the transformer stack and then forwarding the pair downstream so every
  subsequent block can consume the adapter callback. 【F:MultiModalAdapter/clip/model.py†L254-L298】
- `Transformer.forward` is unchanged except for passing through the adapter payload. 【F:MultiModalAdapter/clip/model.py†L217-L234】
- `gradient_scale_layer` defines a lightweight autograd function that scales gradients during
  adapter updates, allowing MMA to balance modality contributions (a capability absent in the
  vanilla CLIP or MVFA/CoOp baselines). 【F:MultiModalAdapter/clip/gsl.py†L3-L16】

Because the adapters are packaged as callables, MVFA-style vision adapters or CoOp-style
prompt learners can coexist: if a framework chooses not to provide an adapter callback, the
CLIP blocks revert to their original execution path with no side effects.

> **External reuse.** When MVFA + CoOp scripts reuse the stock CLIP modules directly (without
> importing `clip/model.py`), the new `coop_prompt_adapter` helpers reproduce the same
> gradient-scaled residual updates around each transformer block. This keeps the integration
> Dassl-free and avoids mutating CoOp's vendor libraries while still matching the behaviour of
> the official MMA trainer.【F:MultiModalAdapter/coop_prompt_adapter.py†L1-L238】

## Trainer behaviour compared to MVFA and CoOp

- **Parameter freezing:** Like MVFA, MMA freezes the pre-trained backbone and only trains the
  newly added modules. When reusing the lightweight wrapper you must explicitly re-enable
  gradients for the adapter learner (`text_adapter`, `visual_adapter`, `shared_adapter`), the
  optional cross-attention bundle, and the shared `logit_scale` while leaving CLIP and Conch
  weights frozen. 【F:train_few_conch1_5_mma.py†L600-L628】
- **Optimization granularity:** Whereas MVFA restricts adapters to the vision branch and CoOp
  learns a small prompt tensor, MMA introduces three coordinated adapter stacks (text, vision,
  shared) plus a learnable gradient scale and cross-attention modules. The integration therefore
  separates optimizer groups for CoOp prompts, Conch detection adapters, MMA bottlenecks, and the
  cross-attention adapters so each component can use an appropriate learning rate and shared
  weight decay. 【F:train_few_conch1_5_mma.py†L642-L679】
- **Forward signature:** MMA augments encoder calls to optionally pass adapter callbacks. For
  compatibility, both `CustomCLIP.encode_*` methods accept `None`, so existing MVFA or CoOp
  code paths can remain untouched if they opt out of MMA. 【F:MultiModalAdapter/clip/mma_custom_clip.py†L131-L139】

## Integration notes for adding MMA to an existing MVFA/CoOp pipeline

1. **Configuration plumbing:** Add the `TRAINER.MMADAPTER` node (prompt prefix, adapter layer
   range, bottleneck dim, scale, precision) to your config schema or reuse `extend_cfg` to
   populate sensible defaults. 【F:MultiModalAdapter/train.py†L88-L97】
2. **Model assembly:** Wrap the shared CLIP backbone with `CustomCLIP`, supply class names,
   and freeze non-adapter parameters before optimizer creation. This keeps MVFA's residual
   visual adapters and CoOp's prompt tokens intact while enabling MMA modules via the adapter
   callbacks. 【F:MultiModalAdapter/clip/mma_custom_clip.py†L82-L139】
3. **Forward integration:** When encoding text/images, pass the adapter functions if MMA is
   enabled; otherwise, forward the tensors as usual. MMA's design ensures the adapters are
   optional and therefore easy to toggle per experiment. 【F:MultiModalAdapter/clip/mma_custom_clip.py†L131-L139】
4. **Gradient management:** If combining with MVFA's residual adapters, coordinate the
   `ADAPTER_SCALE` so the gradient scaling in `gradient_scale_layer` does not overwhelm the
   visual residual path. 【F:MultiModalAdapter/trainers/mmadapter.py†L89-L96】【F:MultiModalAdapter/clip/gsl.py†L3-L16】
5. **Conch-specific routing:** When CLIP's visual stack is replaced by the Conch tower, reuse
   `visual_adapter_func` on the extracted patch tokens (via `_apply_cross_modal_visual_adapters`)
   so both modalities still observe the shared adapters. This mirrors the stock `CustomCLIP`
   forward pass even though the underlying encoder implementation differs. Inspect
   `CONCH_Inplanted_15.det_adapters` to capture the bottleneck width (768 by default) and set
   `mma_cfg.TRAINER.MMADAPTER.VISUAL_WIDTH` accordingly so the adapter bottlenecks and
   cross-attention projections align with the tokens produced after Conch's detection adapters
   compress the ViT-L stream instead of assuming CLIP's native 1024-d width.
6. **High-layer context exchange:** Collect per-layer patch tokens, run
   `compute_mma_text_features` with cross-attention enabled, and feed the returned fused text
   states into `_apply_cross_modal_visual_adapters`. This cross-attention upgrade replaces the
   earlier mean-pooled summaries so each joint layer performs a full token-to-token exchange
   before residual adapters apply their updates.【F:train_few_conch1_5_mma.py†L200-L353】【F:train_few_conch1_5_mma.py†L639-L735】

By keeping the MMA logic encapsulated in `AdapterLearner` and the CLIP fork, the rest of the
training pipeline (data loading, evaluation scripts, logging) remains identical to the CoOp
baseline. This separation is key to integrating MMA cleanly alongside MVFA's vision adapters
and CoOp's prompt learner within a unified framework.
