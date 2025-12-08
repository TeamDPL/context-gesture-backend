#!/usr/bin/env python3
"""
End-to-end multimodal training with raw inputs:
  - Context encoder (OWL-ViT GRU) fine-tune optional
  - Gesture encoder (TD-GCN dual) fine-tune optional
  - Main fused->inventory contrastive + CE
  - Auxiliary context->scene / gesture->intent contrastive (optional if texts provided)

Expected dataset JSONL fields per sample:
  frames: list[str]             # image paths (ordered clip)
  left_seq_22: list             # [T][22][3] left hand joints (pre-normalized 22-joint DHG order)
  right_seq_22: list            # [T][22][3] right hand joints (pre-normalized)
  object_name: str              # canonical label (matches fphab_label_texts.json entry)
  item_embedding: list[float]   # CLIP embedding for fused (affordance+context) text
  context_embedding: list[float]# CLIP embedding for context-only text
  gesture_embedding: list[float]# CLIP embedding for gesture-only text

Notes:
  - If you prefer to freeze encoders, pass --freeze-context / --freeze-gesture.
  - CLIP text encoder remains frozen; only fusion/tree/query/leaf + encoders + aux heads are trained.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from gesture_encoder import TDGCN_Wrist_Encoder
from gesture_context_classifier import (
    FusionTreeConfig,
    GestureContextTreeClassifier,
    SoftDecisionTree,
)
from owl_sgvit_gru import OWLSGVitConfig, OWLSGVitGRU

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_CLIP_PATH = str(REPO_DIR / "hf_models" / "clip-vit-base-patch32")
DEFAULT_OWL_PATH = str(REPO_DIR / "hf_models" / "owlvit-base-patch16")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_label_bank(path: Path) -> Tuple[List[str], torch.Tensor, Dict[str, int]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    names: List[str] = []
    fused: List[List[float]] = []
    for entry in data:
        obj = entry.get("object_name")
        emb = entry.get("fused_embedding")
        if obj is None or emb is None:
            continue
        names.append(str(obj))
        fused.append(list(emb))
    if not fused:
        raise RuntimeError(f"No label embeddings found in {path}")
    fused_tensor = torch.tensor(fused, dtype=torch.float32)
    fused_tensor = F.normalize(fused_tensor, dim=-1)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    return names, fused_tensor, name_to_idx


class RawSequenceDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.samples = load_jsonl(jsonl_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "frames": s["frames"],
            "left_seq_22": s["left_seq_22"],
            "right_seq_22": s["right_seq_22"],
            "object_name": s["object_name"],
            "item_embedding": s["item_embedding"],
            "inventory": s.get("inventory_texts"),
            "target": torch.tensor(int(s["target_idx"]), dtype=torch.long) if "target_idx" in s else None,
            "context_embedding": s.get("context_embedding"),
            "gesture_embedding": s.get("gesture_embedding"),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    targets = [b["target"] for b in batch]
    has_target = all(t is not None for t in targets)

    def stack_embeddings(key: str) -> Optional[torch.Tensor]:
        values = [b[key] for b in batch]
        if not values or any(v is None for v in values):
            return None
        return torch.tensor(values, dtype=torch.float32)

    return {
        "frames": [b["frames"] for b in batch],
        "left_seq_22": torch.tensor([b["left_seq_22"] for b in batch], dtype=torch.float32),
        "right_seq_22": torch.tensor([b["right_seq_22"] for b in batch], dtype=torch.float32),
        "item_embedding": stack_embeddings("item_embedding"),
        "inventory": [b["inventory"] for b in batch],
        "target": torch.stack(targets) if has_target else None,
        "context_embedding": stack_embeddings("context_embedding"),
        "gesture_embedding": stack_embeddings("gesture_embedding"),
        "object_name": [b["object_name"] for b in batch],
    }


class ProjectionHeads(nn.Module):
    def __init__(self, gesture_dim: int, context_dim: int, clip_dim: int):
        super().__init__()
        self.gesture = nn.Linear(gesture_dim, clip_dim)
        self.context = nn.Linear(context_dim, clip_dim)

    def encode_gesture(self, g: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.gesture(g), dim=-1)

    def encode_context(self, c: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.context(c), dim=-1)


def build_fused_query(model: GestureContextTreeClassifier, fused: torch.Tensor, leaf_probs: torch.Tensor) -> torch.Tensor:
    fused_q = F.normalize(model.query_proj(fused), dim=-1)
    if model.cfg.use_leaf_queries:
        leaf_q = F.normalize(model.leaf_queries, dim=-1)
        q = leaf_probs @ leaf_q
        q = F.normalize(q, dim=-1)
    else:
        q = fused_q
    return q


def info_nce(q: torch.Tensor, bank: torch.Tensor, targets: torch.Tensor, temp: float) -> torch.Tensor:
    if bank.numel() == 0:
        return torch.tensor(0.0, device=q.device)
    logits = q @ bank.t() / max(temp, 1e-6)
    return F.cross_entropy(logits, targets)


@dataclass
class TrainingConfig:
    train_path: str = "../dataset/F-PHAB/fphab_processed/fphab_train.jsonl"
    val_path: Optional[str] = None
    batch_size: int = 4
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    log_every: int = 10
    ckpt_dir: str = "checkpoints_end2end"
    ckpt_every: int = 1

    # loss weights/temps
    reg_weight: float = 0.01
    w_fused_ctr: float = 1.0
    w_ctx_ctr: float = 0.3
    w_g_ctr: float = 0.3
    main_temp: float = 0.07
    ctx_temp: float = 0.07
    g_temp: float = 0.07

    # dims
    gesture_dim: int = 512
    context_dim: int = 768
    fusion_dim: int = 256
    tree_depth: int = 3
    max_context_frames: int = 4
    context_max_objects: int = 16
    context_topk_relations: int = 16

    # paths
    clip_local_path: Optional[str] = DEFAULT_CLIP_PATH
    clip_local_files_only: bool = True
    leaf_text_prompts: Optional[List[str]] = None
    owl_model_path: Optional[str] = DEFAULT_OWL_PATH
    owl_processor_path: Optional[str] = DEFAULT_OWL_PATH
    label_texts_path: str = "../dataset/F-PHAB/fphab_label_texts.json"

    # freeze flags
    freeze_context: bool = False
    freeze_gesture: bool = False

    # logging
    use_wandb: bool = True
    wandb_entity: str = "yescher-kyung-hee-university"
    wandb_project: str = "GestSense"
    wandb_run_id: Optional[str] = None


def make_models(cfg: TrainingConfig, device: torch.device):
    # gesture encoder
    gesture_encoder = TDGCN_Wrist_Encoder(device)
    gesture_encoder.train(not cfg.freeze_gesture)
    if cfg.freeze_gesture:
        gesture_encoder.eval()
        for p in gesture_encoder.parameters():
            p.requires_grad_(False)

    # context encoder
    owl_cfg = OWLSGVitConfig(
        text_queries=["object"],
        det_threshold=0.2,
        max_objects=cfg.context_max_objects,
        topk_relations=cfg.context_topk_relations,
        frame_dim=cfg.context_dim,
        use_attn_pool=True,
        owl_pretrained_path=cfg.owl_model_path,
        processor_pretrained_path=cfg.owl_processor_path,
        local_files_only=cfg.clip_local_files_only,
    )
    context_encoder = OWLSGVitGRU(owl_cfg, device=str(device))
    if cfg.freeze_context:
        context_encoder.eval()
        for p in context_encoder.parameters():
            p.requires_grad_(False)
    else:
        context_encoder.train()
        context_encoder.owl.train()

    # classifier
    fusion_cfg = FusionTreeConfig(
        gesture_dim=cfg.gesture_dim,
        context_dim=cfg.context_dim,
        fusion_dim=cfg.fusion_dim,
        tree_depth=cfg.tree_depth,
        use_clip_inventory=True,
        clip_local_path=cfg.clip_local_path,
        clip_local_files_only=cfg.clip_local_files_only,
        use_leaf_queries=True,
        leaf_text_prompts=cfg.leaf_text_prompts,
    )
    classifier = GestureContextTreeClassifier(fusion_cfg, device=str(device))
    if classifier.clip_scorer is not None:
        classifier.clip_scorer.clip.eval()
        for p in classifier.clip_scorer.clip.parameters():
            p.requires_grad_(False)

    proj_heads = ProjectionHeads(cfg.gesture_dim, cfg.context_dim, classifier.clip_scorer.clip_dim).to(device)
    return gesture_encoder, context_encoder, classifier, proj_heads


def encode_gesture_batch(gesture_encoder: TDGCN_Wrist_Encoder, left_seq: torch.Tensor, right_seq: torch.Tensor) -> torch.Tensor:
    # left_seq/right_seq: [B, T, 22, 3]
    left = left_seq.permute(0, 3, 1, 2)   # [B,3,T,22]
    right = right_seq.permute(0, 3, 1, 2) # [B,3,T,22]
    x = torch.stack([right, left], dim=1).unsqueeze(-1)  # [B,2,3,T,22,1]
    return gesture_encoder(x)


def select_context_frames(frame_paths: Sequence[str], max_frames: int) -> List[str]:
    if max_frames <= 0 or len(frame_paths) <= max_frames:
        return list(frame_paths)
    total = len(frame_paths)
    step = total / max_frames
    selected: List[str] = []
    seen = set()
    for i in range(max_frames):
        idx = int(i * step)
        if idx >= total:
            idx = total - 1
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(frame_paths[idx])
    # ensure we always return max_frames (may need to append last frame)
    while len(selected) < max_frames:
        selected.append(frame_paths[-1])
    return selected


def encode_context_batch(
    context_encoder: OWLSGVitGRU,
    frames_batch: Sequence[Sequence[str]],
    max_frames: int,
) -> torch.Tensor:
    outs = []
    for frames in frames_batch:
        limited = select_context_frames(frames, max_frames)
        imgs = [Image.open(f).convert("RGB") for f in limited]
        c = context_encoder.forward_frames(imgs)  # [1, D]
        outs.append(c.squeeze(0))
    return torch.stack(outs, dim=0)


def training_step(
    batch: Dict[str, Any],
    gesture_enc: TDGCN_Wrist_Encoder,
    context_enc: OWLSGVitGRU,
    classifier: GestureContextTreeClassifier,
    proj_heads: ProjectionHeads,
    cfg: TrainingConfig,
    device: torch.device,
    label_bank: torch.Tensor,
    label_name_to_idx: Dict[str, int],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # gesture
    left_seq = batch["left_seq_22"].to(device)
    right_seq = batch["right_seq_22"].to(device)
    g_embed = encode_gesture_batch(gesture_enc, left_seq, right_seq)

    # context
    c_embed = encode_context_batch(context_enc, batch["frames"], cfg.max_context_frames).to(device)

    fused = classifier.fusion(g_embed, c_embed)
    leaf_probs = classifier.tree(fused)

    q = build_fused_query(classifier, fused, leaf_probs)

    indices = torch.tensor(
        [label_name_to_idx.get(name, -1) for name in batch["object_name"]],
        dtype=torch.long,
        device=device,
    )
    if (indices < 0).any():
        missing = [name for name in batch["object_name"] if name not in label_name_to_idx]
        raise ValueError(f"Missing label embeddings for: {missing}")

    logits = q @ label_bank.t() / max(cfg.main_temp, 1e-6)
    targets_ctr = indices
    loss_fused_ctr = F.cross_entropy(logits, targets_ctr)
    loss_reg = cfg.reg_weight * SoftDecisionTree.path_entropy(leaf_probs)

    # aux losses
    context_embeds = batch["context_embedding"]
    gesture_embeds = batch["gesture_embedding"]

    loss_ctx_ctr = torch.tensor(0.0, device=device)
    if context_embeds is not None:
        ctx_txt = F.normalize(context_embeds.to(device), dim=-1)
        ctx_q = proj_heads.encode_context(c_embed)
        ctx_targets = torch.arange(ctx_q.size(0), device=device, dtype=torch.long)
        loss_ctx_ctr = info_nce(ctx_q, ctx_txt, ctx_targets, cfg.ctx_temp)

    loss_g_ctr = torch.tensor(0.0, device=device)
    if gesture_embeds is not None:
        g_txt = F.normalize(gesture_embeds.to(device), dim=-1)
        g_q = proj_heads.encode_gesture(g_embed)
        g_targets = torch.arange(g_q.size(0), device=device, dtype=torch.long)
        loss_g_ctr = info_nce(g_q, g_txt, g_targets, cfg.g_temp)

    loss = loss_reg + cfg.w_fused_ctr * loss_fused_ctr
    loss = loss + cfg.w_ctx_ctr * loss_ctx_ctr + cfg.w_g_ctr * loss_g_ctr

    logs = {
        "loss": float(loss.item()),
        "loss_reg": float(loss_reg.item()),
        "loss_fused_ctr": float(loss_fused_ctr.item()),
        "loss_ctx_ctr": float(loss_ctx_ctr.item()),
        "loss_g_ctr": float(loss_g_ctr.item()),
    }
    return loss, logs


def train(cfg: TrainingConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gesture_enc, context_enc, classifier, proj_heads = make_models(cfg, device)
    label_names, fused_bank, name_to_idx = load_label_bank(Path(cfg.label_texts_path))
    fused_bank = fused_bank.to(device)

    train_ds = RawSequenceDataset(Path(cfg.train_path))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = None
    if cfg.val_path:
        val_ds = RawSequenceDataset(Path(cfg.val_path))
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )

    params: List[dict] = [
        {"params": classifier.fusion.parameters()},
        {"params": classifier.tree.parameters()},
        {"params": classifier.query_proj.parameters()},
        {"params": proj_heads.parameters()},
    ]
    if classifier.cfg.use_leaf_queries:
        params.append({"params": [classifier.leaf_queries], "lr": cfg.lr})
    if not cfg.freeze_gesture:
        params.append({"params": gesture_enc.parameters(), "lr": cfg.lr})
    if not cfg.freeze_context:
        params.append({"params": context_enc.parameters(), "lr": cfg.lr})

    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if cfg.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or disable --use-wandb.")
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=vars(cfg),
            id=cfg.wandb_run_id,
            resume="allow" if cfg.wandb_run_id else None,
        )

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        gesture_enc.train(not cfg.freeze_gesture)
        context_enc.train(not cfg.freeze_context)
        classifier.train()
        proj_heads.train()

        for batch in train_loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss, logs = training_step(
                    batch,
                    gesture_enc,
                    context_enc,
                    classifier,
                    proj_heads,
                    cfg,
                    device,
                    fused_bank,
                    name_to_idx,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if global_step % cfg.log_every == 0:
                print(f"epoch {epoch} step {global_step}: {logs}")
                if wandb_run:
                    wandb_logs = {f"train/{k}": v for k, v in logs.items()}
                    wandb_logs["epoch"] = epoch
                    wandb_logs["step"] = global_step
                    wandb.log(wandb_logs, step=global_step)
            global_step += 1

        if val_loader:
            val_logs = evaluate(
                gesture_enc,
                context_enc,
                classifier,
                proj_heads,
                val_loader,
                cfg,
                device,
                epoch,
                fused_bank,
                name_to_idx,
            )
            if wandb_run:
                wandb_logs = {"epoch": epoch, **val_logs}
                wandb.log(wandb_logs, step=global_step)

        if epoch % cfg.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save({
                "gesture_enc": gesture_enc.state_dict(),
                "classifier": classifier.state_dict(),
                "proj_heads": proj_heads.state_dict(),
                "cfg": cfg.__dict__,
                "context_enc": context_enc.state_dict(),
            }, ckpt_path)
            print(f"saved checkpoint to {ckpt_path}")

    if wandb_run:
        wandb_run.finish()


def evaluate(
    gesture_enc: TDGCN_Wrist_Encoder,
    context_enc: OWLSGVitGRU,
    classifier: GestureContextTreeClassifier,
    proj_heads: ProjectionHeads,
    loader: DataLoader,
    cfg: TrainingConfig,
    device: torch.device,
    epoch: int,
    label_bank: torch.Tensor,
    label_name_to_idx: Dict[str, int],
    progress_desc: Optional[str] = None,
) -> Dict[str, float]:
    gesture_enc.eval()
    context_enc.eval()
    classifier.eval()
    proj_heads.eval()

    total, correct = 0, 0
    total_loss = 0.0
    can_measure_acc = True
    iterator = loader
    pbar = None
    if progress_desc:
        if tqdm is None:
            print("tqdm is not installed; continuing without a progress bar.")
        else:
            total_batches = len(loader)
            iterator = tqdm(loader, desc=progress_desc, total=total_batches, leave=True)
            pbar = iterator
    with torch.no_grad():
        for batch in iterator:
            loss, _ = training_step(
                batch,
                gesture_enc,
                context_enc,
                classifier,
                proj_heads,
                cfg,
                device,
                label_bank,
                label_name_to_idx,
            )
            batch_size = len(batch["frames"])
            total_loss += loss.item() * batch_size

            # accuracy over inventory if provided
            try:
                targets = torch.tensor(
                    [label_name_to_idx[b] for b in batch["object_name"]],
                    dtype=torch.long,
                    device=device,
                )
            except KeyError:
                can_measure_acc = False
                continue

            left_seq = batch["left_seq_22"].to(device)
            right_seq = batch["right_seq_22"].to(device)
            g_embed = encode_gesture_batch(gesture_enc, left_seq, right_seq)
            c_embed = encode_context_batch(context_enc, batch["frames"], cfg.max_context_frames).to(device)

            fused = classifier.fusion(g_embed, c_embed)
            leaf_probs = classifier.tree(fused)
            q = build_fused_query(classifier, fused, leaf_probs)
            logits = q @ label_bank.t() / max(cfg.main_temp, 1e-6)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    if pbar is not None:
        pbar.close()

    denom = len(loader.dataset) if hasattr(loader, "dataset") else max(total, 1)
    avg_loss = total_loss / max(denom, 1)
    logs: Dict[str, float] = {"val/loss": float(avg_loss)}
    if can_measure_acc and total > 0:
        acc = correct / max(total, 1)
        logs["val/acc"] = float(acc)
        print(f"[val] epoch {epoch} loss={avg_loss:.4f} acc={acc:.4f}")
    else:
        print(f"[val] epoch {epoch} loss={avg_loss:.4f} (acc skipped: inventory/target not provided)")
    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train_path", default="../dataset/F-PHAB/fphab_processed/fphab_train.jsonl")
    parser.add_argument("--val", dest="val_path", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-local", type=str, default=DEFAULT_CLIP_PATH)
    parser.add_argument("--no-clip-local-files-only", action="store_true")
    parser.add_argument("--leaf-prompts", type=str, nargs="*", default=None)
    parser.add_argument("--owl-model-path", type=str, default=DEFAULT_OWL_PATH)
    parser.add_argument("--owl-processor-path", type=str, default=DEFAULT_OWL_PATH)
    parser.add_argument("--label-texts", type=str, default="../dataset/F-PHAB/fphab_label_texts.json")
    parser.add_argument("--freeze-context", action="store_true")
    parser.add_argument("--freeze-gesture", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_end2end")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-entity", type=str, default="yescher-kyung-hee-university")
    parser.add_argument("--wandb-project", type=str, default="GestSense")
    parser.add_argument("--wandb-run-id", type=str, default="OWL_GRU_Skel_NoVal_1")
    args = parser.parse_args()

    cfg = TrainingConfig(
        train_path=args.train_path,
        val_path=args.val_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_local_path=args.clip_local,
        clip_local_files_only=not args.no_clip_local_files_only,
        leaf_text_prompts=args.leaf_prompts,
        owl_model_path=args.owl_model_path,
        owl_processor_path=args.owl_processor_path,
        label_texts_path=args.label_texts,
        freeze_context=args.freeze_context,
        freeze_gesture=args.freeze_gesture,
        log_every=args.log_every,
        ckpt_dir=args.ckpt_dir,
        use_wandb=not args.no_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_run_id=args.wandb_run_id,
    )
    train(cfg)
