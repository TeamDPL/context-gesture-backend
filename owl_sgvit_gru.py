import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection


# ---------------------------
# Utils: box -> patch indices
# ---------------------------
def box_to_patch_indices(
    box_xyxy: torch.Tensor,
    img_w: int,
    img_h: int,
    patch_size: int
) -> List[int]:
    """
    Map a bounding box to patch-token indices of ViT grid.
    box_xyxy: (x1, y1, x2, y2) in pixel coords.
    Returns list of patch indices in raster order.
    """
    x1, y1, x2, y2 = box_xyxy.tolist()

    gw, gh = img_w // patch_size, img_h // patch_size  # grid width, height

    px1 = int(max(0, math.floor(x1 / patch_size)))
    py1 = int(max(0, math.floor(y1 / patch_size)))
    px2 = int(min(gw - 1, math.floor(x2 / patch_size)))
    py2 = int(min(gh - 1, math.floor(y2 / patch_size)))

    idxs = []
    for py in range(py1, py2 + 1):
        for px in range(px1, px2 + 1):
            idxs.append(py * gw + px)

    return idxs


# ---------------------------
# Relation Head (SG-ViT style)
# ---------------------------
class RelationHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj_s = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.proj_o = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, obj_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obj_tokens: [M, D]
        returns:
          score: [M, M] scalar relation scores p_ij
          R:     [M, M, D] relation embeddings r_ij
        """
        S = self.proj_s(obj_tokens)                 # [M, D]
        O = self.proj_o(obj_tokens)                 # [M, D]
        score = S @ O.t()                           # [M, M]
        R = self.ln(S[:, None, :] + O[None, :, :])  # [M, M, D]
        return score, R


# ---------------------------
# Attention Pooling over tokens
# ---------------------------
class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim))   # learnable query
        self.proj = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [N, D]
        returns pooled vector: [D]
        """
        # [N, D] -> [N, D]
        k = self.proj(tokens)
        # attention weights: [N]
        attn = torch.softmax(k @ self.q, dim=0)
        # weighted sum -> [D]
        return (attn[:, None] * tokens).sum(dim=0)


# ---------------------------
# Temporal Aggregator (GRU)
# ---------------------------
class GRUAggregator(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: [B, T, D] frame embeddings
        returns context embedding c: [B, D]
        """
        H, _ = self.gru(Z)         # [B, T, hidden]
        h_last = H[:, -1, :]       # last timestep
        c = self.out(h_last)       # project back to D
        return c


# ---------------------------
# Main model wrapper
# ---------------------------
@dataclass
class OWLSGVitConfig:
    text_queries: List[str]
    det_threshold: float = 0.2
    max_objects: int = 64      # M
    topk_relations: int = 32   # K
    frame_dim: int = 768       # OWL-ViT base hidden dim
    use_attn_pool: bool = True
    owl_pretrained_path: Optional[str] = None         # set to local dir to avoid downloads
    processor_pretrained_path: Optional[str] = None   # set to local dir to avoid downloads
    local_files_only: bool = False                    # True for strict offline


class OWLSGVitGRU(nn.Module):
    def __init__(self, config: OWLSGVitConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = config
        self.device = device

        # Resolve paths for offline/local checkpoints
        processor_path = (
            config.processor_pretrained_path
            or os.getenv("OWL_VIT_PROCESSOR_PATH")
            or "google/owlvit-base-patch16"
        )
        model_path = (
            config.owl_pretrained_path
            or os.getenv("OWL_VIT_MODEL_PATH")
            or "google/owlvit-base-patch16"
        )

        # HF OWL-ViT
        self.processor = OwlViTProcessor.from_pretrained(
            processor_path,
            local_files_only=config.local_files_only,
        )
        self.owl = OwlViTForObjectDetection.from_pretrained(
            model_path,
            local_files_only=config.local_files_only,
        ).to(device)
        self.owl.eval()  # we start inference mode

        # relation head + pooling + temporal
        self.rel_head = RelationHead(config.frame_dim).to(device)
        self.pool = AttnPool(config.frame_dim).to(device) if config.use_attn_pool else None
        self.temporal = GRUAggregator(config.frame_dim).to(device)

    ()
    def _extract_vision_tokens(self, outputs) -> torch.Tensor:
        """
        Grab the last hidden states from the vision tower across HF versions.
        """
        if getattr(outputs, "vision_model_output", None) is not None:
            return outputs.vision_model_output.last_hidden_state[0]
        if getattr(outputs, "vision_hidden_states", None):
            return outputs.vision_hidden_states[-1][0]
        if getattr(outputs, "vision_outputs", None) is not None:
            return outputs.vision_outputs.last_hidden_state[0]
        raise AttributeError("OWL-ViT output is missing vision hidden states")

    ()
    def forward_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """
        frames: list of PIL Images length T
        return context embedding c: [1, D]
        """
        z_list = []
        for img in frames:
            z_t = self.encode_one_frame(img)  # [D]
            z_list.append(z_t)

        Z = torch.stack(z_list, dim=0).unsqueeze(0)  # [1, T, D]
        c = self.temporal(Z)                         # [1, D]
        return c

    ()
    def encode_one_frame(self, image: Image.Image) -> torch.Tensor:
        """
        Per-frame pipeline:
          OWL-ViT detect -> boxes
          patch pooling -> obj embeddings
          relation head -> topK relation tokens
          pool -> z_t
        returns z_t: [D]
        """
        image = image.convert("RGB")
        text = [self.cfg.text_queries]  # HF expects batch nesting

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.owl(**inputs, output_hidden_states=True)
        vision_tokens = self._extract_vision_tokens(outputs)

        # -------------------------
        # 1) Boxes from OWL-ViT
        # -------------------------
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (h,w)
        det = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.cfg.det_threshold
        )[0]

        boxes = det["boxes"]  # [M',4]
        scores = det["scores"]

        if boxes.numel() == 0:
            # fallback: no detections -> use CLS-like global token pool
            return vision_tokens.mean(dim=0)

        # keep top max_objects by score
        order = torch.argsort(scores, descending=True)
        order = order[: self.cfg.max_objects]
        boxes = boxes[order]

        # -------------------------
        # 2) Patch tokens
        # -------------------------
        img_w, img_h = image.size
        patch_size = self.processor.image_processor.patch_size  # usually 16

        # -------------------------
        # 3) Box-based patch pooling -> obj embeddings
        # -------------------------
        obj_embeds = []
        for b in boxes:
            idxs = box_to_patch_indices(b, img_w, img_h, patch_size)
            if len(idxs) == 0:
                continue
            tok = vision_tokens[idxs, :]    # [n_patches, D]
            obj_embeds.append(tok.mean(dim=0))

        if len(obj_embeds) == 0:
            return vision_tokens.mean(dim=0)

        obj_tokens = torch.stack(obj_embeds, dim=0)  # [M, D]

        # -------------------------
        # 4) Relation head
        # -------------------------
        score, R = self.rel_head(obj_tokens)  # score [M,M], R [M,M,D]

        # -------------------------
        # 5) Top-K relation tokens
        # -------------------------
        M, _, D = R.shape
        flat_score = score.flatten()                # [M*M]
        K = min(self.cfg.topk_relations, flat_score.numel())
        topk_idx = flat_score.topk(K).indices       # [K]
        rel_tokens = R.view(-1, D)[topk_idx]        # [K, D]

        # -------------------------
        # 6) Frame embedding z_t
        # -------------------------
        tokens = torch.cat([obj_tokens, rel_tokens], dim=0)  # [(M+K), D]
        if self.pool is None:
            z_t = tokens.mean(dim=0)
        else:
            z_t = self.pool(tokens)

        return z_t


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    cfg = OWLSGVitConfig(
        text_queries=["object"],    # Trick: extract all the objects whatever they are
        det_threshold=0.2,
        max_objects=64,
        topk_relations=32,
        frame_dim=768,      # owlvit-base hidden dim
        use_attn_pool=True,
        owl_pretrained_path="./hf_models/owlvit-base-patch16",         
        processor_pretrained_path="./hf_models/owlvit-base-patch16",   
        local_files_only=True,     
    )

    model = OWLSGVitGRU(cfg, device="cuda" if torch.cuda.is_available() else "cpu")

    # load a short clip (T frames)
    frames = [Image.open(f"frames/{i:03d}.jpg") for i in range(8)]  # ex: 8 frames / 2s

    c = model.forward_frames(frames)  # [1, D]
    print("context embedding:", c.shape)
