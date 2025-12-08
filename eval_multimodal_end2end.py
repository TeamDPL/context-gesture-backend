#!/usr/bin/env python3
"""
Evaluate a trained multimodal model checkpoint on a JSONL dataset.
Usage example:
  python eval_multimodal_end2end.py \
      --test ../dataset/F-PHAB/fphab_processed/fphab_test.jsonl \
      --ckpt checkpoints_end2end/epoch_5.pt \
      --label-texts ../dataset/F-PHAB/fphab_label_texts.json
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from train_multimodal_end2end import (
    TrainingConfig,
    make_models,
    RawSequenceDataset,
    collate_fn,
    load_label_bank,
    evaluate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a multimodal checkpoint on a JSONL dataset.")
    parser.add_argument("--test", type=str, default="../dataset/F-PHAB/fphab_processed/fphab_test.jsonl", help="Path to JSONL file (e.g., test split).")
    parser.add_argument("--ckpt", type=str, default="./checkpoints_end2end/epoch_6.pt", help="Checkpoint path saved by train_multimodal_end2end.py.")
    parser.add_argument("--label-texts", type=str, default="../dataset/F-PHAB/fphab_label_texts.json", help="Override path to fphab_label_texts.json.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict: Dict[str, Any] = ckpt.get("cfg")
    if cfg_dict is None:
        raise RuntimeError(f"Checkpoint {ckpt_path} missing 'cfg' dictionary.")
    cfg = TrainingConfig(**cfg_dict)
    cfg.train_path = args.test  # not used but keeps reference
    if args.label_texts:
        cfg.label_texts_path = args.label_texts

    dataset = RawSequenceDataset(Path(args.test))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gesture_enc, context_enc, classifier, proj_heads = make_models(cfg, device)
    gesture_enc.load_state_dict(ckpt["gesture_enc"])
    context_enc.load_state_dict(ckpt["context_enc"])
    classifier.load_state_dict(ckpt["classifier"])
    proj_heads.load_state_dict(ckpt["proj_heads"])

    label_names, fused_bank, name_to_idx = load_label_bank(Path(cfg.label_texts_path))
    fused_bank = fused_bank.to(device)

    logs = evaluate(
        gesture_enc,
        context_enc,
        classifier,
        proj_heads,
        loader,
        cfg,
        device,
        epoch=0,
        label_bank=fused_bank,
        label_name_to_idx=name_to_idx,
    )
    print("Evaluation complete:", logs)


if __name__ == "__main__":
    main()
