import os
import argparse
import random
from types import SimpleNamespace
from contextlib import nullcontext

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import timm
from datetime import datetime
from dataset.medical_few import MedDataset
from CLIP.clip import create_model as create_clip
from conch.open_clip_custom.adapter_15 import CONCH_Inplanted_15

from MultiModalAdapter.coop_prompt_learner import PromptLearner, TextEncoder
from MultiModalAdapter.clip.mma_custom_clip import CustomCLIP
from MultiModalAdapter.coop_prompt_adapter import (
    CrossAttentionBundle,
    encode_prompts_with_cross_attention,
    encode_prompts_with_layer_context,
)
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, accuracy_score, f1_score
from utils import augment, cos_sim
from prompt import REAL_NAME
from loss import FocalLoss, BinaryDiceLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

CLASS_INDEX = {
    "Brain": 3,
    "Liver": 2,
    "Retina_RESC": 1,
    "Retina_OCT2017": -1,
    "Chest": -2,
    "Histopathology": -3,
    "CRC": -4,
    'SICAP':-5,
    'LCLCC':-7,
}

VISION_TO_TEXT_MAP = {
    6: 3,
    12: 6,
    18: 9,
    24: 12,
}

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_clip_backbone(args):
    clip_model = create_clip(
        model_name=args.model_name,
        img_size=args.img_size,
        pretrained=args.pretrain,
        precision=args.coop_precision,
        device=device,
        require_pretrained=True,
    )
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model

def build_conch_encoder():
    ckpt_path = "./conch/checkpoints/pytorch_model_vision.bin"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"CONCH v1.5 checkpoint not found: {ckpt_path}. "
            "Place `pytorch_model_vision.bin` there before running HAAF."
        )
    state_dict = torch.load(ckpt_path, map_location="cpu")

    vision_tower = timm.create_model(
        "vit_large_patch16_224",
        img_size=448,
        patch_size=16,
        init_values=1.0,
        num_classes=0,
        dynamic_img_size=True,
    )

    prefix = "trunk."
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state[k[len(prefix):]] = v
        else:
            new_state[k] = v

    missing, unexpected = vision_tower.load_state_dict(new_state, strict=False)
    print("[Conch] missing:", missing, "| unexpected:", unexpected)

    vision_tower.eval()
    for p in vision_tower.parameters():
        p.requires_grad = False
    return vision_tower

def build_binary_classnames(dataset_key: str):
    base_name = REAL_NAME[dataset_key]
    return [f"normal {base_name}", f"abnormal {base_name}"]

def build_coop_cfg(args):
    return SimpleNamespace(
        TRAINER=SimpleNamespace(
            COOP=SimpleNamespace(
                N_CTX=args.coop_n_ctx,
                CTX_INIT=args.coop_ctx_init,
                CSC=args.coop_csc,
                CLASS_TOKEN_POSITION=args.coop_class_pos,
            )
        )
    )

def build_prompt_learner(args, classnames, clip_model):
    coop_cfg = build_coop_cfg(args)
    prompt_learner = PromptLearner(
        coop_cfg, classnames, clip_model, device=device
    ).to(device)
    prompt_learner.train()
    for p in prompt_learner.parameters():
        p.requires_grad = True
    return prompt_learner

def compute_coop_text_features(prompt_learner, text_encoder, precision="fp32", detach=False):
    amp = precision == "fp16" and torch.cuda.is_available()
    ctx = torch.cuda.amp.autocast(enabled=amp, dtype=torch.float16) if amp else nullcontext()
    grad_ctx = torch.no_grad() if detach else nullcontext()

    tokenized = prompt_learner.tokenized_prompts
    with grad_ctx:
        prompts = prompt_learner()
        with ctx:
            text_features, _ = encode_prompts_with_layer_context(
                text_encoder, prompts, tokenized_prompts=tokenized
            )

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.t().contiguous()
    if detach:
        text_features = text_features.detach()
    return text_features

def evaluate_coop_text_features(prompt_learner, text_encoder, precision="fp32"):
    was_training = prompt_learner.training
    prompt_learner.eval()
    with torch.no_grad():
        feats = compute_coop_text_features(prompt_learner, text_encoder, precision, detach=True)
    prompt_learner.train(was_training)
    return feats

def infer_conch_token_width(conch_detector: nn.Module):
    adapters = getattr(conch_detector, "det_adapters", None)
    if adapters is None or len(adapters) == 0:
        return getattr(conch_detector, "embed_dim", None)

    sample_adapter = adapters[0]
    fc1 = getattr(sample_adapter, "fc1", None)

    if isinstance(fc1, nn.Linear):
        return fc1.out_features
    if isinstance(fc1, nn.Sequential):
        for layer in fc1:
            if isinstance(layer, nn.Linear):
                return layer.out_features
    return getattr(conch_detector, "embed_dim", None)

def _initialise_cross_attention_adapters(mma_model, clip_model, joint_layers, vision_width_override=None):
    if mma_model is None or not joint_layers:
        return []

    text_layer_count = len(clip_model.transformer.resblocks)
    valid_layers = [l for l in sorted(set(joint_layers)) if 1 <= l <= text_layer_count]
    if not valid_layers:
        mma_model.cross_attention_adapters = None
        return []

    text_width = getattr(clip_model.transformer, "width", None)
    if text_width is None:
        text_width = clip_model.ln_final.weight.shape[0]

    attn = clip_model.transformer.resblocks[0].attn
    num_heads = getattr(attn, "num_heads", 8)

    if vision_width_override is not None:
        vision_width = int(vision_width_override)
    else:
        ln_post = getattr(getattr(clip_model, "visual", None), "ln_post", None)
        if ln_post is not None and getattr(ln_post, "weight", None) is not None:
            vision_width = ln_post.weight.shape[0]
        else:
            vision_width = text_width

    mma_model.cross_attention_adapters = CrossAttentionBundle(
        text_width,
        num_heads,
        valid_layers,
        vision_width,
    )
    return valid_layers

def init_mma_model(args, classnames, clip_model, coop_text_encoder, conch_backbone, conch_width):
    mma_cfg = SimpleNamespace(
        INPUT=SimpleNamespace(SIZE=(args.img_size, args.img_size)),
        TRAINER=SimpleNamespace(
            MMADAPTER=SimpleNamespace(
                TEXT_CTX_INIT=args.mma_text_init,
                ADAPTER_DIM=args.mma_adapter_dim,
                ADAPTER_SCALE=args.mma_text_adapter_scale,
                VISUAL_WIDTH=int(conch_width) if conch_width is not None else None,
                JOINT_LAYERS=args.mma_joint_layers,
            )
        ),
    )

    mma_model = CustomCLIP(mma_cfg, classnames, clip_model).to(device)
    mma_model.text_encoder = coop_text_encoder
    mma_model.image_encoder = conch_backbone

    valid_layers = _initialise_cross_attention_adapters(
        mma_model,
        clip_model,
        joint_layers=args.mma_joint_layers,
        vision_width_override=conch_width,
    )
    if getattr(mma_model, "cross_attention_adapters", None) is not None:
        mma_model.cross_attention_adapters = mma_model.cross_attention_adapters.to(device)
    else:
        print("[MMA] warning: no valid joint layers for cross-attention, bundle is None.")

    adapter_learner = getattr(mma_model, "adapter_learner", None)
    if adapter_learner is not None:
        def text_func_no_shared(layer_idx: int):
            seq_adp = adapter_learner.text_adapter.get(layer_idx)
            return seq_adp, None, adapter_learner.adapter_scale

        adapter_learner.text_adapter_func = text_func_no_shared

        if hasattr(adapter_learner, "visual_adapter") and adapter_learner.visual_adapter is not None:
            def vis_func_no_shared(layer_idx: int):
                seq_adp = adapter_learner.visual_adapter.get(layer_idx)
                return seq_adp, None, adapter_learner.adapter_scale
            adapter_learner.visual_adapter_func = vis_func_no_shared
        else:
            def vis_func_empty(_layer_idx: int):
                return None, None, 1.0
            adapter_learner.visual_adapter_func = vis_func_empty

    args.mma_active_layers = list(valid_layers)
    for p in mma_model.parameters():
        p.requires_grad = False

    if adapter_learner is not None:
        for p in adapter_learner.text_adapter.parameters():
            p.requires_grad = True
        if getattr(adapter_learner, "visual_adapter", None) is not None:
            for p in adapter_learner.visual_adapter.parameters():
                p.requires_grad = True

    ca_bundle = getattr(mma_model, "cross_attention_adapters", None)
    if ca_bundle is not None:
        for p in ca_bundle.parameters():
            p.requires_grad = True

    if isinstance(getattr(mma_model, "logit_scale", None), nn.Parameter):
        mma_model.logit_scale.requires_grad = True

    mma_model.train()
    return mma_model

def extract_patch_token_map(det_patch_tokens, feature_indices, v2t_map: dict):
    out = {}
    for tokens, v_layer in zip(det_patch_tokens, feature_indices):
        t_layer = v2t_map.get(v_layer, None)
        if t_layer is not None:
            out[t_layer] = tokens
    return out

def apply_mma_on_text(
    prompt_learner,
    coop_text_encoder,
    mma_model,
    patch_token_map,
    joint_layers,
    precision="fp32",
    detach=False,
    return_layer_states=False,
    mma_text_fusion_scale=0.1,
):
    if mma_model is None:
        raise ValueError("MMA model is required for MMA text features")

    adapter_learner = getattr(mma_model, "adapter_learner", None)
    cross_attention = getattr(mma_model, "cross_attention_adapters", None)

    text_dev = next(coop_text_encoder.parameters()).device

    patch_token_map_on_text = {
        layer_idx: tokens.to(text_dev)
        for layer_idx, tokens in patch_token_map.items()
    }

    if cross_attention is not None:
        cross_attention = cross_attention.to(text_dev)

    amp = precision == "fp16" and torch.cuda.is_available()
    ctx = (
        torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
        if amp
        else nullcontext()
    )
    grad_ctx = torch.no_grad() if detach else nullcontext()

    tokenized = prompt_learner.tokenized_prompts

    with grad_ctx:
        prompts = prompt_learner()
        with ctx:
            text_embeddings, layer_states = encode_prompts_with_cross_attention(
                text_encoder=coop_text_encoder,
                prompts=prompts,
                tokenized_prompts=tokenized,
                adapter_func=getattr(adapter_learner, "text_adapter_func", None),
                patch_token_map=patch_token_map_on_text,
                cross_attention=cross_attention,
                joint_layers=joint_layers,
                mma_text_fusion_scale=mma_text_fusion_scale,
            )

    text_features = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    text_features = text_features.t().contiguous()

    if detach:
        text_features = text_features.detach()
        layer_states = {k: v.detach() for k, v in layer_states.items()}

    if return_layer_states:
        return text_features, layer_states
    return text_features

def apply_mma_on_patches(
    det_patch_tokens,
    mma_model,
    feature_indices,
    joint_layers,
    text_states=None,
    mma_scale=0.1,
    v2t_map: dict = None,
):
    if mma_model is None:
        return det_patch_tokens

    ca_bundle = getattr(mma_model, "cross_attention_adapters", None)
    out_tokens = []
    joint_layer_set = set(joint_layers or [])

    for tokens, v_layer in zip(det_patch_tokens, feature_indices):
        new_tokens = tokens
        t_layer = v2t_map.get(v_layer, None) if v2t_map is not None else v_layer

        if (
            t_layer in joint_layer_set
            and ca_bundle is not None
            and text_states is not None
            and t_layer in text_states
        ):
            adapter = ca_bundle.get_t_to_v(t_layer)
            if adapter is not None:
                txt_state = text_states[t_layer]
                txt_state = txt_state.mean(dim=0, keepdim=True)
                vis_query = new_tokens.unsqueeze(0)
                txt_state = txt_state.to(vis_query.device, vis_query.dtype)

                fused = adapter(vis_query, txt_state).squeeze(0)
                new_tokens = new_tokens + mma_scale * (fused - new_tokens)

        out_tokens.append(new_tokens)

    return out_tokens

def build_losses():
    return {
        "focal": FocalLoss(),
        "dice": BinaryDiceLoss(),
        "bce": nn.BCEWithLogitsLoss(),
    }

def train_epoch(
    args,
    model_det,
    mma_model,
    prompt_learner,
    coop_text_encoder,
    clip_model,
    legacy_text_features,
    baseline_logit_scale,
    train_loader,
    optimizer,
):
    model_det.train()
    if mma_model is not None:
        mma_model.train()
    if prompt_learner is not None:
        prompt_learner.train()

    loss_bce = nn.BCEWithLogitsLoss()
    losses = []

    use_coop = prompt_learner is not None
    use_mma = mma_model is not None and args.enable_mma

    mma_text_layers = getattr(args, "mma_active_layers", args.mma_joint_layers)

    for image, _, label in tqdm(train_loader, desc="Train"):
        image = image.to(device)
        label = label.to(device)

        det_feats = model_det(image)
        det_tokens = [f[0, 1:, :] for f in det_feats]

        patch_token_map = extract_patch_token_map(
            det_tokens, args.features_list, VISION_TO_TEXT_MAP
        )
        patch_token_map_for_text = {
            layer: tokens.detach() for layer, tokens in patch_token_map.items()
        }
        if use_coop:
            if use_mma:
                text_features, text_states = apply_mma_on_text(
                    prompt_learner,
                    coop_text_encoder,
                    mma_model,
                    patch_token_map_for_text,
                    mma_text_layers,
                    precision=args.coop_precision,
                    detach=False,
                    return_layer_states=True,
                    mma_text_fusion_scale=args.mma_text_fusion_scale,
                )
                logit_scale = mma_model.logit_scale.exp()
            else:
                text_features = compute_coop_text_features(
                    prompt_learner, coop_text_encoder, precision=args.coop_precision
                )
                text_states = {}
                logit_scale = baseline_logit_scale
        else:
            text_features = legacy_text_features
            text_states = {}
            logit_scale = baseline_logit_scale

        if use_mma:
            det_tokens = apply_mma_on_patches(
                det_tokens,
                mma_model,
                args.features_list,
                joint_layers=mma_text_layers,
                text_states=text_states,
                v2t_map=VISION_TO_TEXT_MAP,
                mma_scale=args.mma_vision_fusion_scale
            )

        feat_dtype = det_tokens[0].dtype
        if text_features.dtype != feat_dtype:
            text_features = text_features.to(feat_dtype)
        logit_scale = logit_scale.to(feat_dtype)

        total_det_loss = 0.0
        for tok in det_tokens:
            tok = tok / tok.norm(dim=-1, keepdim=True)
            logits = logit_scale * (tok @ text_features)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            img_prob = prob.mean().unsqueeze(0)
            target = label.float().view_as(img_prob)
            total_det_loss += loss_bce(img_prob, target)

        optimizer.zero_grad()
        total_det_loss.backward()
        optimizer.step()
        losses.append(total_det_loss.item())

    return float(np.mean(losses))

@torch.no_grad()
def evaluate(
    args,
    model_det,
    mma_model,
    prompt_learner,
    coop_text_encoder,
    clip_model,
    legacy_text_features,
    baseline_logit_scale,
    train_loader,
    test_loader,
):
    model_det.eval()
    if mma_model is not None:
        mma_model.eval()
    if prompt_learner is not None:
        prompt_learner.eval()

    use_coop = prompt_learner is not None
    use_mma = mma_model is not None and args.enable_mma
    mma_text_layers = getattr(args, "mma_active_layers", args.mma_joint_layers)

    print("Building prototypes...")
    normal_feats_per_layer = [[] for _ in args.features_list]
    abnormal_feats_per_layer = [[] for _ in args.features_list]

    for image, _, label in tqdm(train_loader, desc="Build Proto"):
        image = image.to(device)
        label = label.to(device)

        det_feats = model_det(image)
        det_tokens = [f[0, 1:, :] for f in det_feats]
        patch_token_map = extract_patch_token_map(
            det_tokens, args.features_list, VISION_TO_TEXT_MAP
        )

        text_states = {}
        if use_mma and use_coop:
            _, text_states = apply_mma_on_text(
                prompt_learner,
                coop_text_encoder,
                mma_model,
                patch_token_map,
                mma_text_layers,
                precision=args.coop_precision,
                detach=True,
                return_layer_states=True,
                mma_text_fusion_scale=args.mma_text_fusion_scale,
            )

        final_det_tokens = det_tokens
        if use_mma:
            final_det_tokens = apply_mma_on_patches(
                det_tokens,
                mma_model,
                args.features_list,
                joint_layers=mma_text_layers,
                text_states=text_states,
                v2t_map=VISION_TO_TEXT_MAP,
                mma_scale=args.mma_vision_fusion_scale
            )

        target_list = abnormal_feats_per_layer if label.item() == 1 else normal_feats_per_layer
        for i, tok in enumerate(final_det_tokens):
            target_list[i].append(tok)

    proto_normal_per_layer = [
        torch.mean(torch.cat(feats, dim=0), dim=0, keepdim=True)
        for feats in normal_feats_per_layer
    ]
    proto_abnormal_per_layer = [
        torch.mean(torch.cat(feats, dim=0), dim=0, keepdim=True)
        for feats in abnormal_feats_per_layer
    ]
    print("Prototypes built.")

    if use_mma:
        text_features_static = None
        logit_scale = mma_model.logit_scale.exp().to(device)
    elif use_coop:
        text_features_static = evaluate_coop_text_features(
            prompt_learner, coop_text_encoder, precision=args.coop_precision
        ).to(device)
        logit_scale = clip_model.logit_scale.exp().to(device)
    else:
        text_features_static = legacy_text_features.to(device)
        logit_scale = baseline_logit_scale.to(device)

    gt = []
    scores_zero = []
    scores_proto = []

    for image, y, _ in tqdm(test_loader, desc="Test"):
        image = image.to(device)
        gt.append(y.item())

        det_feats = model_det(image)
        det_tokens = [f[0, 1:, :] for f in det_feats]
        patch_token_map = extract_patch_token_map(
            det_tokens, args.features_list, VISION_TO_TEXT_MAP
        )

        if use_mma and use_coop:
            text_features, text_states = apply_mma_on_text(
                prompt_learner,
                coop_text_encoder,
                mma_model,
                patch_token_map,
                mma_text_layers,
                precision=args.coop_precision,
                detach=True,
                return_layer_states=True,
                mma_text_fusion_scale=args.mma_text_fusion_scale,
            )
            text_features = text_features.to(device)
        else:
            text_states = {}
            text_features = text_features_static

        if use_mma:
            det_tokens = apply_mma_on_patches(
                det_tokens,
                mma_model,
                args.features_list,
                joint_layers=mma_text_layers,
                text_states=text_states,
                v2t_map=VISION_TO_TEXT_MAP,
                mma_scale=args.mma_vision_fusion_scale
            )

        feat_dtype = det_tokens[0].dtype
        if text_features.dtype != feat_dtype:
            text_features = text_features.to(feat_dtype)
        this_logit_scale = logit_scale.to(feat_dtype)

        total_dist_normal = 0.0
        total_dist_abnormal = 0.0

        for layer_idx, tok in enumerate(det_tokens):
            p_norm = proto_normal_per_layer[layer_idx].to(tok.device, tok.dtype)
            p_abnorm = proto_abnormal_per_layer[layer_idx].to(tok.device, tok.dtype)

            tok = tok / tok.norm(dim=-1, keepdim=True)
            p_norm = p_norm / p_norm.norm(dim=-1, keepdim=True)
            p_abnorm = p_abnorm / p_abnorm.norm(dim=-1, keepdim=True)

            dist_norm = (1 - (tok @ p_norm.T)).mean()
            dist_abnormal = (1 - (tok @ p_abnorm.T)).mean()

            total_dist_normal += dist_norm
            total_dist_abnormal += dist_abnormal

        score_p = total_dist_normal / (total_dist_normal + total_dist_abnormal + 1e-8)
        scores_proto.append(score_p.item())

        score_zs = 0.0
        for tok in det_tokens:
            logits = this_logit_scale * (tok @ text_features)
            prob_abn = torch.softmax(logits, dim=-1)[:, 1]
            score_zs += float(prob_abn.mean().cpu().numpy())
        scores_zero.append(score_zs)

    gt = np.array(gt)
    scores_zero = np.array(scores_zero)
    scores_proto = np.array(scores_proto)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    scores_zero = norm(scores_zero)
    scores_proto = norm(scores_proto)

    zero_auc = roc_auc_score(gt, scores_zero)
    proto_auc = roc_auc_score(gt, scores_proto)
    print("zero auc", zero_auc, " proto auc", proto_auc)
    final_scores = 0.5 * scores_zero + 0.5 * scores_proto
    auc = roc_auc_score(gt, final_scores)

    ap = average_precision_score(gt, final_scores)

    precisions, recalls, thresholds = precision_recall_curve(gt, final_scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]

    best_thresh = thresholds[min(best_f1_idx, len(thresholds)-1)]
    pred_labels = (final_scores >= best_thresh).astype(int)
    acc = accuracy_score(gt, pred_labels)

    metrics = {
        "auc": auc,
        "ap": ap,
        "f1_max": best_f1,
        "acc": acc
    }

    return metrics

def parse_args():
    p = argparse.ArgumentParser(description="Few-shot Medical AD with MMA")
    p.add_argument("--device", type=str, default=None, help="Device, e.g. cuda, cuda:0, or cpu.")
    p.add_argument("--model_name", type=str, default="ViT-L-14-336")
    p.add_argument("--pretrain", type=str, default="openai")
    p.add_argument("--obj", type=str, default="Histopathology")
    p.add_argument("--data_path", type=str, default="./data/")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--save_model", type=int, default=1)
    p.add_argument("--save_path", type=str, default="./ckpt/few_mma_gemini/")
    p.add_argument("--img_size", type=int, default=240)
    p.add_argument("--epoch", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--shot", type=int, default=4)
    p.add_argument("--iterate", type=int, default=-1)
    p.add_argument("--coop-n-ctx", type=int, default=16)
    p.add_argument("--coop-ctx-init", type=str, default="")
    p.add_argument("--coop-class-pos", type=str, default="end", choices=["end", "middle"])
    p.add_argument("--coop-csc", default=True)
    p.add_argument("--coop-precision", type=str, default="fp32", choices=["fp16", "fp32"])
    p.add_argument("--disable-coop-prompts", action="store_true")
    p.add_argument("--coop-prompt-lr", type=float, default=1e-4)
    mma_group = p.add_mutually_exclusive_group()
    mma_group.add_argument("--enable-mma", dest="enable_mma", action="store_true")
    mma_group.add_argument("--disable-mma", dest="enable_mma", action="store_false")
    p.set_defaults(enable_mma=True)
    p.add_argument("--mma-text-init", type=str, default="")
    p.add_argument("--mma-adapter-dim", type=int, default=128)
    p.add_argument("--mma_text_adapter_scale", type=float, default=0.2)
    p.add_argument("--mma_vision_fusion_scale", type=float, default=1)
    p.add_argument("--mma_text_fusion_scale", type=float, default=1)
    p.add_argument("--mma-joint-layers", type=int, nargs="+", default=[3, 6, 9, 12])
    p.add_argument("--mma-adapter-lr", type=float, default=1e-5)
    p.add_argument("--mma-cross-attn-lr", type=float, default=1e-4)
    return p.parse_args()

def main():
    global device, use_cuda
    args = parse_args()
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    setup_seed(args.seed)

    clip_model = build_clip_backbone(args)
    coop_text_encoder = TextEncoder(clip_model).to(device)
    for p in coop_text_encoder.parameters():
        p.requires_grad = False

    classnames = build_binary_classnames(args.obj)
    conch_backbone = build_conch_encoder()

    det_model = CONCH_Inplanted_15(
        conch_v15_model=conch_backbone,
        features=args.features_list,
    ).to(device)

    for p in det_model.parameters():
        p.requires_grad = False
    for p in det_model.det_adapters.parameters():
        p.requires_grad = True

    conch_width = infer_conch_token_width(det_model)
    mma_model = None
    if args.enable_mma:
        mma_model = init_mma_model(
            args,
            classnames,
            clip_model,
            coop_text_encoder,
            conch_backbone,
            conch_width,
        )

    prompt_learner = None
    legacy_text_features = None
    use_coop = not args.disable_coop_prompts

    
    prompt_learner = build_prompt_learner(args, classnames, clip_model)
   

    baseline_logit_scale = clip_model.logit_scale.exp().detach()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )

    aug_abn_img, aug_abn_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    aug_norm_img, aug_norm_mask = augment(test_dataset.fewshot_norm_img)

    fewshot_img = torch.cat([aug_abn_img, aug_norm_img], dim=0)
    fewshot_mask = torch.cat([aug_abn_mask, aug_norm_mask], dim=0)
    fewshot_label = torch.cat(
        [torch.ones(len(aug_abn_img)), torch.zeros(len(aug_norm_img))],
        dim=0,
    )

    train_dataset = torch.utils.data.TensorDataset(fewshot_img, fewshot_mask, fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    support_dataset = torch.utils.data.TensorDataset(aug_norm_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)

    param_groups = []

    det_params = [p for p in det_model.det_adapters.parameters() if p.requires_grad]
    if det_params:
        param_groups.append({"params": det_params, "lr": args.learning_rate})

    if prompt_learner is not None:
        coop_params = [p for p in prompt_learner.parameters() if p.requires_grad]
        if coop_params:
            param_groups.append({"params": coop_params, "lr": args.coop_prompt_lr})

    if mma_model is not None:
        adapter_learner = getattr(mma_model, "adapter_learner", None)
        if adapter_learner is not None:
            for name in ("text_adapter", "shared_adapter"):
                cont = getattr(adapter_learner, name, None)
                if cont is not None:
                    mma_params = [p for p in cont.parameters() if p.requires_grad]
                    if mma_params:
                        param_groups.append({
                            "params": mma_params,
                            "lr": args.mma_adapter_lr,
                        })

        ca_bundle = getattr(mma_model, "cross_attention_adapters", None)
        if ca_bundle is not None:
            ca_params = [p for p in ca_bundle.parameters() if p.requires_grad]
            if ca_params:
                param_groups.append({
                    "params": ca_params,
                    "lr": args.mma_cross_attn_lr,
                })

        if isinstance(getattr(mma_model, "logit_scale", None), nn.Parameter):
            param_groups.append({
                "params": [mma_model.logit_scale],
                "lr": args.mma_adapter_lr,
            })

    optimizer = torch.optim.Adam(param_groups, betas=(0.5, 0.999), weight_decay=1e-4)

    best_auc = 0.0
    os.makedirs(args.save_path, exist_ok=True)
    print("text layers:", len(clip_model.transformer.resblocks))

    for epoch in range(args.epoch):
        print(f"\nEpoch {epoch}:")
        loss = train_epoch(
            args,
            det_model,
            mma_model,
            prompt_learner,
            coop_text_encoder,
            clip_model,
            legacy_text_features,
            baseline_logit_scale,
            train_loader,
            optimizer,
        )
        print(f"  Train loss: {loss:.4f}")

        metrics = evaluate(
            args,
            det_model,
            mma_model,
            prompt_learner,
            coop_text_encoder,
            clip_model,
            legacy_text_features,
            baseline_logit_scale,
            train_loader,
            test_loader,
        )
        
        auc = metrics["auc"]
        ap = metrics["ap"]
        f1 = metrics["f1_max"]
        acc = metrics["acc"]

        print(f"  {args.obj} Results - AUC: {auc:.4f} | AP: {ap:.4f} | F1: {f1:.4f} | ACC: {acc:.4f}")

        is_best = False
        if auc > best_auc:
            best_auc = auc
            is_best = True

        result_path = f'./result/FINAL/{args.obj}.txt'
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_message = f"[{timestamp}] AUC: {auc:.4f} | AP: {ap:.4f} | F1: {f1:.4f} | ACC: {acc:.4f}\n"

        if is_best:
            log_message += f"  (*** New Best AUC: {best_auc:.4f} ***)\n"
            print(f"\n*** [{timestamp}] New Best AUC for {args.obj}: {best_auc:.4f} (Epoch {epoch}) ***\n")
        else:
            log_message += "\n"

        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(log_message)

        if args.save_model and is_best:
            ckpt = {
                "det_adapters": det_model.det_adapters.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_auc": best_auc,
                "best_metrics": metrics,
            }
            if mma_model is not None:
                ckpt["mma_adapter"] = mma_model.state_dict()
            if prompt_learner is not None:
                ckpt["prompt_learner"] = prompt_learner.state_dict()

            out_path = os.path.join(args.save_path, f"{args.obj}.pth")
            torch.save(ckpt, out_path)

if __name__ == "__main__":
    main()
