# Integrating the Multi-Modal Adapter into `train_few_conch1_5_coop.py`

This guide extends the MVFA + CoOp baseline implemented in
`train_few_conch1_5_coop.py` by adding the Multi-Modal Adapter (MMA) components
introduced in the `MultiModalAdapter` package. Follow the steps sequentially to
keep the script clean and debuggable. Each step references the exact helpers and
modules you will touch so you can cross-check with the source when something
breaks.

## 0. Prepare the working copy

1. Duplicate the baseline script before making edits. For example:
   ```bash
   cp train_few_conch1_5_coop.py train_few_conch1_5_mma.py
   ```
   Working on a copy makes it easy to diff against the original when you need to
   verify the MMA-specific changes.
2. Ensure the `MultiModalAdapter` directory lives alongside the script (it is
   already present in this repository). The guide assumes all relative imports
   can be resolved from `PROJECT_ROOT` just like the existing CoOp references.

## 1. Import the MMA building blocks

Add the following near the other top-level imports so that the adapter learner,
prompt bridge, and CLIP wrapper are available without mutating `sys.path`:

```python
from MultiModalAdapter.coop_prompt_learner import PromptLearner, TextEncoder
from MultiModalAdapter.clip.mma_custom_clip import CustomCLIP
from MultiModalAdapter.coop_prompt_adapter import (
    CrossAttentionBundle,
    encode_prompts_with_cross_attention,
    encode_prompts_with_layer_context,
)
```

`coop_prompt_learner` mirrors CoOp's prompt utilities but sources CLIP
dependencies from the shared `CLIP` package (notably
`CLIP.tokenizer.SimpleTokenizer` and `CLIP.tokenizer.tokenize`), allowing MMA to
reuse the exact prompt logic
without depending on the `CoOp-main` repository. The adapter and encoding
helpers continue to bridge MMA callbacks into the shared text encoder so the
vanilla CoOp script stays untouched.【F:MultiModalAdapter/coop_prompt_learner.py†L1-L127】【F:MultiModalAdapter/clip/mma_custom_clip.py†L68-L139】【F:MultiModalAdapter/coop_prompt_adapter.py†L1-L94】

## 2. Extend the CLI with MMA flags

Keep the MMA knobs next to the existing CoOp argument group so users can toggle
between prompt-only and adapter-enhanced runs. Add a new argument group:

```python
mma_group = parser.add_argument_group("MMA adapters")
mma_group.add_argument("--enable-mma", action="store_true", help="Activate Multi-Modal Adapters.")
mma_group.add_argument("--mma-text-init", type=str, default="a medical",
                       help="Prefix string used to seed MMA text prompts.")
mma_group.add_argument("--mma-adapter-start", type=int, default=0,
                       help="Transformer block index where adapters become active.")
mma_group.add_argument("--mma-adapter-end", type=int, default=23,
                       help="Last transformer block index that owns an MMA adapter.")
mma_group.add_argument("--mma-adapter-dim", type=int, default=512,
                       help="Bottleneck width for MMA adapters.")
mma_group.add_argument("--mma-adapter-scale", type=float, default=1.0,
                       help="Residual scaling applied after each adapter.")
mma_group.add_argument("--mma-joint-layers", type=int, nargs="+", default=[18, 24],
                       help="Transformer blocks that exchange high-level context across vision/text.")
```

The defaults above mirror the configuration fields consumed by
`AdapterLearner` (`TEXT_CTX_INIT`, `ADAPTER_START/END/DIM/SCALE`).【F:MultiModalAdapter/clip/mma_custom_clip.py†L82-L124】

## 3. Build a minimal MMA config namespace

`AdapterLearner` expects a Dassl-style config object with nested attributes. Reuse
Python's `SimpleNamespace` (already imported) to construct the required shape
after parsing arguments:

```python
mma_cfg = SimpleNamespace(
    INPUT=SimpleNamespace(SIZE=(args.img_size, args.img_size)),
    TRAINER=SimpleNamespace(
        MMADAPTER=SimpleNamespace(
            TEXT_CTX_INIT=args.mma_text_init,
            ADAPTER_START=args.mma_adapter_start,
            ADAPTER_END=args.mma_adapter_end,
            ADAPTER_DIM=args.mma_adapter_dim,
            ADAPTER_SCALE=args.mma_adapter_scale,
        )
    ),
)
```

This mirrors the structure created by `train.py` inside the MMA package and
provides only the fields that `AdapterLearner` reads.

## 4. Register MMA adapters around the existing encoders

Right after freezing the vanilla CLIP backbone and before the CoOp prompt
learner is created, branch on `args.enable_mma`. Construct `CustomCLIP` so that
its adapter learner can be reused, but immediately point both encoder handles to
the **already-instantiated** components: the CoOp text tower (with learnable
tokens) and the Conch vision backbone. No additional encoder instances should be
constructed.

```python
mma_model = None
if args.enable_mma:
    classnames = build_binary_classnames(args.obj)
    mma_model = CustomCLIP(mma_cfg, classnames, clip_model).to(device)
    mma_model.text_encoder = coop_text_encoder      # share CoOp text encoder + tokens
    mma_model.image_encoder = conch_v15_model       # share Conch vision encoder
    mma_model.train()
    for param in mma_model.parameters():
        param.requires_grad = False

    adapter_learner = getattr(mma_model, "adapter_learner", None)
    if adapter_learner is not None:
        for container_name in ("text_adapter", "visual_adapter", "shared_adapter"):
            container = getattr(adapter_learner, container_name, None)
            if container is None:
                continue
            for param in container.parameters():
                param.requires_grad = True

    cross_attention = getattr(mma_model, "cross_attention_adapters", None)
    if cross_attention is not None:
        for param in cross_attention.parameters():
            param.requires_grad = True

    if isinstance(mma_model.logit_scale, torch.nn.Parameter):
        mma_model.logit_scale.requires_grad = True
```

`CustomCLIP` exposes the adapter stacks via `adapter_learner`, but the actual
forward pass must rely on the CoOp prompt learner to create prompts (with
learnable tokens) and on `coop_text_encoder` to translate those prompts into
text features. The adapters enrich those shared features at higher layers; they
never replace the prompt learner or spawn an independent text encoder. Likewise,
repointing `mma_model.image_encoder` keeps the Conch feature extractor in play
so the visual adapters wrap the same activations MVFA already uses.
【F:MultiModalAdapter/clip/mma_custom_clip.py†L68-L139】【F:MultiModalAdapter/coop_prompt_learner.py†L33-L134】【F:conch/open_clip_custom/adapter_15.py†L25-L72】

## 5. Manage optimizer groups

Create four parameter groups so each learnable component receives an appropriate
learning rate while keeping the rest of CLIP and Conch frozen:

```python
optimizer_groups = []

if use_coop_prompts:
    optimizer_groups.append({
        "params": [p for p in prompt_learner.parameters() if p.requires_grad],
        "lr": args.coop_prompt_lr,
    })

optimizer_groups.append({
    "params": [p for p in model.det_adapters.parameters() if p.requires_grad],
    "lr": args.learning_rate,
})

mma_bottleneck = []
mma_cross_attention = []

if args.enable_mma and mma_model is not None:
    adapter_learner = mma_model.adapter_learner
    for name in ("text_adapter", "visual_adapter", "shared_adapter"):
        container = getattr(adapter_learner, name, None)
        if container is not None:
            mma_bottleneck.extend(p for p in container.parameters() if p.requires_grad)

    if isinstance(mma_model.logit_scale, torch.nn.Parameter) and mma_model.logit_scale.requires_grad:
        mma_bottleneck.append(mma_model.logit_scale)

    cross_attention = getattr(mma_model, "cross_attention_adapters", None)
    if cross_attention is not None:
        mma_cross_attention.extend(p for p in cross_attention.parameters() if p.requires_grad)

if mma_bottleneck:
    optimizer_groups.append({"params": mma_bottleneck, "lr": args.learning_rate})

if mma_cross_attention:
    optimizer_groups.append({
        "params": mma_cross_attention,
        "lr": args.learning_rate * 5.0,
    })

det_optimizer = torch.optim.Adam(
    optimizer_groups,
    betas=(0.5, 0.999),
    weight_decay=1e-4,
)
```

The cross-attention adapters start from scratch, so they benefit from the
higher learning rate. The shared bottleneck adapters and logit scale stick to
the MVFA step size, while CoOp prompts continue to use their dedicated
`args.coop_prompt_lr`.

## 6. Keep CoOp in charge of text features during training

The prompt learner remains the sole producer of text features. MMA injects
residuals around the shared CLIP transformer, but CoOp still defines the
prompts and token order. The helper now returns optional layer summaries so the
vision branch can influence the text stack at specific transformer blocks:

```python
def compute_mma_text_features(..., layer_bias=None, hook_layers=None, return_layer_states=False):
    prompt_adapter = _build_mma_prompt_adapter(mma_model.adapter_learner)
    return _encode_prompts_with_text_encoder(
        prompt_learner,
        coop_text_encoder,
        precision=precision,
        detach=detach,
        prompt_adapter=prompt_adapter,
        adapter_func=mma_model.adapter_learner.text_adapter_func,
        layer_bias=layer_bias,
        hook_layers=hook_layers,
        capture_layer_states=return_layer_states,
    )
```

`coop_prompt_adapter` now mirrors MMA's adapter residual computations in pure
PyTorch, replaying the gradient-scaled bottleneck math after each CLIP
transformer block. This keeps the integration compatible with the unmodified
CoOp text encoder while still honouring MMA's adapter weights and scaling
rules.【F:MultiModalAdapter/coop_prompt_adapter.py†L41-L238】

When the visual backbone does not share CLIP's hidden width (e.g. the Conch
encoder emits 768-d tokens after its detection adapters compress ViT-L's
1024-d stream), populate `mma_cfg.TRAINER.MMADAPTER.VISUAL_WIDTH` with the
*token* dimensionality that MMA will actually see. The training script now
inspects `CONCH_Inplanted_15.det_adapters` to derive the bottleneck width and
writes that value into both the config namespace and the cross-attention
bundle. This ensures the adapter bottlenecks and multi-head projections align
with the patch-token tensors observed during fusion, eliminating the previous
shape mismatch.

During training, build a patch-token lookup table for the requested
`--mma-joint-layers`, pass it into the new cross-attention aware helper, and
reuse the returned layer states when driving the vision adapters:

```python
text_features, text_states = compute_mma_text_features(
    prompt_learner,
    coop_text_encoder,
    mma_model,
    _extract_patch_token_map(det_patch_tokens, args.features_list),
    mma_visual_layers,
    precision=args.coop_precision,
    return_layer_states=True,
)
det_patch_tokens = _apply_cross_modal_visual_adapters(
    det_patch_tokens,
    mma_model.adapter_learner,
    args.features_list,
    joint_layers=mma_visual_layers,
    text_states=text_states,
    cross_attention=mma_model.cross_attention_adapters,
)
```

CoOp therefore remains the source of truth for prompts and tokenisation, while
MMA consumes and enriches those tensors. The new helpers expose the hidden
states needed to weave high-level vision cues back into the text transformer
and then reflect the fused prompts into the visual branch through
cross-attention.

## 7. Recompute text features for each evaluation image

Evaluation still relies on CoOp to generate prompts, but MMA now injects
image-specific context at the configured joint layers. Switch both the adapter
learner and prompt learner to eval mode, then rebuild the text features inside
the test loop using the same summary helpers introduced in Step 6:

```python
if enable_mma and mma_model is not None:
    mma_prev_mode = mma_model.training
    mma_model.eval()
    prompt_eval_guard = prompt_learner.training
    prompt_learner.train(False)
    logit_scale = mma_model.logit_scale.exp().detach()

for image, _, _ in test_loader:
    det_patch_tokens = model(image)
    det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

    text_states = {}
    if enable_mma and mma_model is not None and use_coop_prompts:
        text_feats, text_states = compute_mma_text_features(
            prompt_learner,
            coop_text_encoder,
            mma_model,
            _extract_patch_token_map(det_patch_tokens, args.features_list),
            mma_visual_layers,
            precision=coop_precision,
            detach=True,
            return_layer_states=True,
        )
    else:
        text_feats = text_features_static  # CoOp or legacy branch

    det_patch_tokens = _apply_cross_modal_visual_adapters(
        det_patch_tokens,
        mma_model.adapter_learner,
        args.features_list,
        joint_layers=mma_visual_layers,
        text_states=text_states,
        cross_attention=mma_model.cross_attention_adapters,
    )
```

After processing the dataloader, restore `prompt_learner.train(prompt_eval_guard)`
and `mma_model.train(mma_prev_mode)` so subsequent training epochs continue
unaffected. Zero-shot and few-shot scoring reuse `text_feats` exactly like the
baseline; the difference is that those features now embed per-image vision cues
before normalisation.

## 8. Inject MMA visual adapters into MVFA tokens

`CustomCLIP` exposes `visual_adapter_func` so every ViT block can receive the
same residual injection that enriches the text tower. Reuse those adapters by
wrapping the patch tokens emitted from `CONCH_Inplanted_15` with a helper that
mirrors the reference MMA residual math:

```python
def _apply_cross_modal_visual_adapters(
    det_patch_tokens,
    adapter_learner,
    feature_ids,
    joint_layers=None,
    text_states=None,
    cross_attention=None,
):
    if adapter_learner is None:
        return det_patch_tokens
    visual_adapter_func = getattr(adapter_learner, "visual_adapter_func", None)
    if visual_adapter_func is None:
        return det_patch_tokens

    joint_layer_set = set(joint_layers or [])
    text_states = text_states or {}

    adapted = []
    for tokens, layer_idx in zip(det_patch_tokens, feature_ids):
        seq_adapter, shared_adapter, scale = visual_adapter_func(layer_idx)
        if layer_idx in joint_layer_set and cross_attention is not None:
            fused_text = text_states.get(layer_idx)
            adapter = cross_attention.get_t_to_v(layer_idx)
            if fused_text is not None and adapter is not None:
                tokens = adapter(tokens, fused_text.to(tokens.dtype))

        if seq_adapter is not None:
            if hasattr(seq_adapter, "down") and hasattr(seq_adapter, "up"):
                residual = seq_adapter.down(tokens)
                if shared_adapter is not None:
                    residual = shared_adapter(residual)
                residual = seq_adapter.up(residual)
            else:
                residual = seq_adapter(tokens)
                if shared_adapter is not None:
                    residual = shared_adapter(residual)
            tokens = tokens + scale * residual.to(tokens.dtype)

        adapted.append(tokens)
    return adapted
```

Apply the helper wherever patch tokens are produced: immediately after the
Conch forward pass in training, while constructing the memory bank, and inside
`test()` before computing few-shot cosine distances or zero-shot logits. Pass
the `text_states` derived in Step 6/7 so the visual adapters receive the same
cross-attended prompts as the text branch, matching the dual-modality behaviour
implemented in the upgraded MMA forward path.

## 9. Save and resume MMA checkpoints

Augment the checkpoint dictionaries with MMA state so runs can be resumed:

```python
if args.save_model == 1:
    checkpoint = {
        "det_adapters": model.det_adapters.state_dict(),
        "optimizer": det_optimizer.state_dict(),
    }
    if args.enable_mma and mma_model is not None:
        checkpoint["mma_adapter"] = mma_model.state_dict()
    if use_coop_prompts and prompt_learner is not None:
        checkpoint["prompt_learner"] = prompt_learner.state_dict()
    torch.save(checkpoint, ckp_path)
```

On resume, load `mma_adapter` before rebuilding optimizers so parameter groups
contain the restored tensors.

For backward compatibility, wrap the load in a `checkpoint.get("mma_adapter")`
check. When weights are missing or partially compatible, print the `missing`
and `unexpected` lists returned by `load_state_dict(strict=False)` so you can
quickly diagnose mismatched adapter spans or bottleneck sizes.

## 10. Debugging checklist

- **Shape mismatches:** Confirm `args.img_size` matches the CLIP resolution. The
  adapter builder asserts this internally. If the assertion in
  `AdapterLearner.__init__` fires, inspect `mma_cfg.INPUT.SIZE` and
  `clip_model.visual.input_resolution`.
- **Gradient flow:** Print the norms of `mma_model.adapter_learner.text_adapter`
  parameters during training. If they stay zero, ensure they were added to the
  optimizer groups and that `requires_grad` was set correctly in Step 4.
- **Mixed precision:** MMA infers both text and vision dtypes from the shared
  CLIP weights, matching the precision used by CoOp and MVFA. If you enable
  fp16 prompts (`--coop-precision fp16`), double-check that images and adapters
  are cast to the same dtype before computing logits to avoid NaNs.
- **Device alignment:** The adapter learner now moves tokenized prompts onto the
  CLIP token-embedding device prior to lookup, avoiding CPU↔GPU mismatches when
  the backbone is loaded on CUDA.

## Should we add a repository-level `AGENTS.md`?

The current instructions are contained entirely within this guide, and no
repository-wide automation hooks are required. Adding an `AGENTS.md` is
therefore unnecessary at this time. If future tasks require automated style
constraints, create one under the relevant directory tree.
