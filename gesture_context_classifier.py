"""
gesture_context_classifier.py (paper-aligned refactor)

Paper-aligned runtime:
  1) Gesture encoder -> z_g
  2) Context encoder -> z_c
  3) FusionBlock(z_g, z_c) -> fused f
  4) SoftDecisionTree(f) -> leaf_probs p(l | f)
  5) Build CLIP-space query q(f) by mixing learnable leaf queries:
        q = sum_l p_l * q_l
     (optionally you can use fused-query directly)
  6) Encode per-sample inventory texts with CLIP text encoder -> a_i
  7) Item logits = cosine_similarity(q, a_i)
  8) Predict argmax over items in that sample's inventory

So there is NO action/class label head here. Inventory texts are the classes.

This matches the paper's intent:
- fused embedding is aligned to affordance text space during training (contrastive loss)
- at runtime, fused selects items via similarity in that space with soft tree gating.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, CLIPModel
except ImportError:
    AutoTokenizer = None
    CLIPModel = None


# -------------------------
# Config
# -------------------------
@dataclass
class FusionTreeConfig:
    gesture_dim: int
    context_dim: int
    fusion_dim: int = 256
    tree_depth: int = 3
    temperature: float = 0.7
    dropout: float = 0.1

    # CLIP inventory scoring
    use_clip_inventory: bool = True
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_local_path: Optional[str] = None  # set to a local folder to avoid downloads
    clip_local_files_only: bool = False    # set True for strict offline mode

    # Leaf query mixing (paper style)
    use_leaf_queries: bool = True
    leaf_text_prompts: Optional[List[str]] = None
    # If leaf_text_prompts is None, leaf queries are learned params only.

    # Optional scaling/temperature for similarity
    sim_temperature: float = 0.07  # typical CLIP temp; smaller => sharper


# -------------------------
# Fusion block (same as before)
# -------------------------
class FusionBlock(nn.Module):
    def __init__(self, gesture_dim: int, context_dim: int, fusion_dim: int, dropout: float = 0.1):
        super().__init__()
        self.g_proj = nn.Linear(gesture_dim, fusion_dim)
        self.c_proj = nn.Linear(context_dim, fusion_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(fusion_dim * 4),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, gesture: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if gesture.dim() == 1:
            gesture = gesture.unsqueeze(0)
        if context.dim() == 1:
            context = context.unsqueeze(0)

        g = self.g_proj(gesture)
        c = self.c_proj(context)
        joint = torch.cat([g, c, torch.abs(g - c), g * c], dim=-1)
        return self.out(joint)


# -------------------------
# Soft decision tree
# -------------------------
class SoftDecisionTree(nn.Module):
    def __init__(self, input_dim: int, depth: int, temperature: float = 0.7):
        super().__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.temperature = temperature

        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth

        self.decision = nn.Parameter(torch.randn(self.n_internal, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(self.n_internal))

        self.paths = self._build_paths()

    def _build_paths(self) -> List[List[Tuple[int, int]]]:
        paths: List[List[Tuple[int, int]]] = []
        for leaf in range(self.n_leaves):
            node = 0
            leaf_path: List[Tuple[int, int]] = []
            for level in range(self.depth):
                go_right = (leaf >> (self.depth - 1 - level)) & 1
                leaf_path.append((node, go_right))
                node = node * 2 + 1 + go_right
            paths.append(leaf_path)
        return paths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns leaf_probs only.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        gate_logits = F.linear(x, self.decision, self.bias)  # [B, n_internal]
        gate_probs = torch.sigmoid(gate_logits / self.temperature)

        leaf_probs: List[torch.Tensor] = []
        for path in self.paths:
            prob = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
            for node_idx, go_right in path:
                gate = gate_probs[:, node_idx]
                prob = prob * gate if go_right else prob * (1.0 - gate)
            leaf_probs.append(prob)

        leaf_probs = torch.stack(leaf_probs, dim=-1)  # [B, n_leaves]
        leaf_probs = leaf_probs / (leaf_probs.sum(dim=-1, keepdim=True) + 1e-9)
        return leaf_probs

    @staticmethod
    def path_entropy(leaf_probs: torch.Tensor) -> torch.Tensor:
        probs = leaf_probs.clamp_min(1e-6)
        return (-probs * probs.log()).sum(dim=-1).mean()


# -------------------------
# CLIP Inventory scorer
# -------------------------
class ClipInventoryScorer(nn.Module):
    """
    Encodes texts with CLIP text encoder.
    Provides:
      - clip_dim
      - text encoder forward
      - optional textual initialization for leaf queries
    """
    def __init__(
        self,
        model_name: str,
        device: str,
        local_path: Optional[str] = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        if CLIPModel is None or AutoTokenizer is None:
            raise ImportError("transformers[torch] required for CLIP inventory scoring.")

        self.device = device
        # prefer caller path -> env -> model name
        resolved_path = local_path or os.getenv("CLIP_MODEL_PATH") or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path, local_files_only=local_files_only)
        self.clip = CLIPModel.from_pretrained(resolved_path, local_files_only=local_files_only).to(device).eval()
        self.clip_dim = self.clip.config.projection_dim  # e.g., 512

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list[str]
        returns: [N, D_clip] L2-normalized
        """
        if len(texts) == 0:
            return torch.zeros((0, self.clip_dim), device=self.device)

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        feats = self.clip.get_text_features(**tokens)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def batch_encode_inventory(
        self, inventory_batch: Sequence[Sequence[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inventory_batch: B x variable-length list[str]
        returns:
          inv_embeds_padded: [B, I_max, D_clip]
          inv_mask:          [B, I_max]  (True where valid)
        """
        if len(inventory_batch) == 0:
            raise ValueError("Empty inventory_batch")

        if isinstance(inventory_batch[0], str):  # type: ignore
            inventory_batch = [inventory_batch]  # batch size 1

        embeds_list = []
        lens = []
        for inv in inventory_batch:
            emb = self.encode_texts(list(inv))  # [I, D]
            embeds_list.append(emb)
            lens.append(emb.size(0))

        I_max = max(lens)
        B = len(embeds_list)
        D = self.clip_dim

        inv_embeds = torch.zeros((B, I_max, D), device=self.device)
        inv_mask = torch.zeros((B, I_max), device=self.device, dtype=torch.bool)

        for b, emb in enumerate(embeds_list):
            i_len = emb.size(0)
            if i_len > 0:
                inv_embeds[b, :i_len] = emb
                inv_mask[b, :i_len] = True

        return inv_embeds, inv_mask


# -------------------------
# Main classifier (inventory = classes)
# -------------------------
class GestureContextTreeClassifier(nn.Module):
    def __init__(self, config: FusionTreeConfig, device: Optional[str] = None):
        super().__init__()
        self.cfg = config
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Core
        self.fusion = FusionBlock(config.gesture_dim, config.context_dim, config.fusion_dim, config.dropout)
        self.tree = SoftDecisionTree(config.fusion_dim, config.tree_depth, config.temperature)

        # CLIP scorer
        self.clip_scorer = None
        if config.use_clip_inventory:
            self.clip_scorer = ClipInventoryScorer(
                config.clip_model_name,
                self.device_name,
                local_path=config.clip_local_path,
                local_files_only=config.clip_local_files_only,
            )
            clip_dim = self.clip_scorer.clip_dim

            # fused -> clip query
            self.query_proj = nn.Linear(config.fusion_dim, clip_dim)

            # leaf queries (learned)
            if config.use_leaf_queries:
                self.leaf_queries = nn.Parameter(torch.randn(self.tree.n_leaves, clip_dim) * 0.02)

                # optional textual init of leaf queries
                if config.leaf_text_prompts is not None:
                    with torch.no_grad():
                        leaf_txt = self.clip_scorer.encode_texts(config.leaf_text_prompts)  # [L_txt, D]
                    L = self.tree.n_leaves
                    if leaf_txt.size(0) < L:
                        # repeat if fewer prompts than leaves
                        leaf_txt = leaf_txt.repeat((L + leaf_txt.size(0) - 1) // leaf_txt.size(0), 1)[:L]
                    else:
                        leaf_txt = leaf_txt[:L]
                    self.leaf_queries.data.copy_(leaf_txt)

        self.to(self.device_name)

    def forward(
        self,
        gesture_embed: torch.Tensor,
        context_embed: torch.Tensor,
        inventory_texts: Optional[Sequence[Sequence[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          item_logits: [B, I_max] (padded), invalid items = -inf
          leaf_probs:  [B, L]
          inv_mask:    [B, I_max]
        """
        gesture_embed = gesture_embed.to(self.device_name)
        context_embed = context_embed.to(self.device_name)

        fused = self.fusion(gesture_embed, context_embed)  # [B, fusion_dim]
        leaf_probs = self.tree(fused)                      # [B, L]

        if self.clip_scorer is None:
            raise RuntimeError("use_clip_inventory=False is not supported in paper-aligned runtime. Enable CLIP.")

        if inventory_texts is None:
            raise RuntimeError("inventory_texts required for CLIP inventory scoring.")

        inv_embeds, inv_mask = self.clip_scorer.batch_encode_inventory(inventory_texts)  # [B,I,D], [B,I]

        # Make query in CLIP space
        fused_q = F.normalize(self.query_proj(fused), dim=-1)  # [B, D]

        if self.cfg.use_leaf_queries:
            # leaf mixing to produce final query
            leaf_q = F.normalize(self.leaf_queries, dim=-1)    # [L, D]
            q = leaf_probs @ leaf_q                            # [B, D]
            q = F.normalize(q, dim=-1)
        else:
            # simple fused query
            q = fused_q

        # Similarity to inventory items
        # item_logits[b,i] = (q_b · inv_embeds_bi) / sim_temperature
        item_logits = torch.einsum("bd,bid->bi", q, inv_embeds)
        item_logits = item_logits / max(self.cfg.sim_temperature, 1e-6)

        # mask invalid items
        item_logits = item_logits.masked_fill(~inv_mask, float("-inf"))

        return item_logits, leaf_probs, inv_mask

    def compute_loss(
        self,
        gesture_embed: torch.Tensor,
        context_embed: torch.Tensor,
        targets: torch.Tensor,
        inventory_texts: Sequence[Sequence[str]],
        reg_weight: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        targets: [B] each is index into that sample's inventory list.
                 (so targets must be consistent with inventory_texts[b])
        """
        item_logits, leaf_probs, inv_mask = self.forward(
            gesture_embed,
            context_embed,
            inventory_texts=inventory_texts,
        )
        targets = targets.to(item_logits.device)

        ce = F.cross_entropy(item_logits, targets)
        entropy_penalty = SoftDecisionTree.path_entropy(leaf_probs)
        loss = ce + reg_weight * entropy_penalty

        return loss, {
            "loss": float(loss.item()),
            "ce": float(ce.item()),
            "path_entropy": float(entropy_penalty.item()),
        }

    @torch.no_grad()
    def predict(
        self,
        gesture_embed: torch.Tensor,
        context_embed: torch.Tensor,
        inventory_texts: Sequence[Sequence[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          pred_indices: [B] predicted index in each sample inventory
          item_logits:  [B, I_max]
        """
        item_logits, leaf_probs, inv_mask = self.forward(
            gesture_embed,
            context_embed,
            inventory_texts=inventory_texts,
        )
        pred_idx = torch.argmax(item_logits, dim=-1)
        return pred_idx, item_logits


# -------------------------
# Usage sanity sketch
# -------------------------
if __name__ == "__main__":
    # Dummy check
    cfg = FusionTreeConfig(
        gesture_dim=128,
        context_dim=768,
        fusion_dim=256,
        tree_depth=3,
        use_clip_inventory=True,
        clip_local_path="./hf_models/clip-vit-base-patch32",
        clip_local_files_only=True,  
        use_leaf_queries=True,
        leaf_text_prompts=[
            # IMPORTANT: replace with meaningful leaf prompts later
            "combat situation, choose weapon",
            "cooking situation, choose kitchen tool",
            "crafting situation, choose building tool",
        ],
    )

    model = GestureContextTreeClassifier(cfg)

    B = 2
    g = torch.randn(B, cfg.gesture_dim)
    c = torch.randn(B, cfg.context_dim)

    inventory = [
        ["knife", "cutting board", "pot"],
        ["sauce pan", "white bowl"],
    ]
    # targets are per-sample indices into the above lists
    y = torch.tensor([0, 1])

    loss, logs = model.compute_loss(g, c, y, inventory_texts=inventory)
    loss.backward()
    print("sanity logs:", logs)

    pred_idx, logits = model.predict(g, c, inventory)
    print("pred idx:", pred_idx)
    print("logits shape:", logits.shape)
