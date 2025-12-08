import os
import json
import base64
import datetime
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from gesture_context_classifier import FusionTreeConfig, GestureContextTreeClassifier
from gesture_encoder import GestureStreamProcessor
from owl_sgvit_gru import OWLSGVitConfig, OWLSGVitGRU


# ==== Checkpoint / Paths ====
REPO_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = REPO_DIR / "checkpoints_end2end" / "epoch_6.pt"
CKPT_PATH = Path(os.getenv("E2E_CKPT_PATH", str(DEFAULT_CKPT)))

# ==== Globals ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI()

# runtime caches
latest_gesture_embed = None  # torch.Tensor [1, G]
latest_context_embed = None  # torch.Tensor [1, C]
is_left_hand_wrist_based = False


def _resolve_path(from_cfg, env_var, default_subdir):
    """
    Resolve a local HF model path.
    Priority: explicit cfg -> env var -> repo default -> None
    Returns (path_or_none, local_files_only_flag)
    """
    # 1) explicit from cfg / checkpoint
    if from_cfg:
        p = Path(from_cfg).expanduser()
        if p.exists():
            return str(p), True

    # 2) environment override
    env_val = os.getenv(env_var)
    if env_val:
        p = Path(env_val).expanduser()
        if p.exists():
            return str(p), True

    # 3) repo-local default
    repo_path = REPO_DIR / default_subdir
    if repo_path.exists():
        return str(repo_path), True

    # fallback: allow download
    return None, False


def _load_models():
    """
    Load end-to-end checkpoint (gesture encoder + context encoder + classifier)
    and rebuild configs using saved hyperparameters where available.
    """
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(str(CKPT_PATH), map_location=device)
    ck_cfg = ckpt.get("cfg", {}) or {}

    # Resolve paths (safe fallback if original absolute paths are invalid)
    clip_path, clip_local_only = _resolve_path(
        ck_cfg.get("clip_local_path"), "CLIP_MODEL_PATH", "hf_models/clip-vit-base-patch32"
    )
    owl_path, owl_local_only = _resolve_path(
        ck_cfg.get("owl_model_path"), "OWL_VIT_MODEL_PATH", "hf_models/owlvit-base-patch16"
    )
    owl_proc_path, owl_proc_local_only = _resolve_path(
        ck_cfg.get("owl_processor_path"), "OWL_VIT_PROCESSOR_PATH", "hf_models/owlvit-base-patch16"
    )

    # local_files_only flag: keep True only when we actually have the files
    clip_local_only = clip_local_only and clip_path is not None
    owl_local_only = owl_local_only and owl_path is not None
    owl_proc_local_only = owl_proc_local_only and owl_proc_path is not None

    # Build fusion/classifier config
    fusion_cfg = FusionTreeConfig(
        gesture_dim=ck_cfg.get("gesture_dim", 512),
        context_dim=ck_cfg.get("context_dim", 768),
        fusion_dim=ck_cfg.get("fusion_dim", 256),
        tree_depth=ck_cfg.get("tree_depth", 3),
        use_clip_inventory=True,
        clip_local_path=clip_path,
        clip_local_files_only=clip_local_only,
        use_leaf_queries=True,
        leaf_text_prompts=ck_cfg.get("leaf_text_prompts"),
    )

    # Build context encoder config (OWL-ViT GRU)
    owl_cfg = OWLSGVitConfig(
        text_queries=["object"],
        det_threshold=ck_cfg.get("det_threshold", 0.2),
        max_objects=ck_cfg.get("context_max_objects", 16),
        topk_relations=ck_cfg.get("context_topk_relations", 16),
        frame_dim=ck_cfg.get("context_dim", 768),
        use_attn_pool=True,
        owl_pretrained_path=owl_path,
        processor_pretrained_path=owl_proc_path,
        local_files_only=owl_local_only and owl_proc_local_only,
    )

    # Instantiate models
    context_model = OWLSGVitGRU(owl_cfg, device=str(device))
    classifier = GestureContextTreeClassifier(fusion_cfg, device=str(device))
    gesture_processor = GestureStreamProcessor(device=device)

    # Load weights (non-strict to allow minor key mismatches)
    if "context_enc" in ckpt:
        context_model.load_state_dict(ckpt["context_enc"], strict=False)
    if "classifier" in ckpt:
        classifier.load_state_dict(ckpt["classifier"], strict=False)
    if "gesture_enc" in ckpt:
        gesture_processor.encoder.load_state_dict(ckpt["gesture_enc"], strict=False)

    # Eval mode for inference
    context_model.eval()
    classifier.eval()
    gesture_processor.encoder.eval()

    # Frame buffer for context aggregation
    max_frames = ck_cfg.get("max_context_frames", 4)
    frame_buffer = deque(maxlen=max_frames if max_frames and max_frames > 0 else 4)

    return gesture_processor, context_model, classifier, frame_buffer, ck_cfg


gesture_processor, context_model, classifier, frame_buffer, ck_cfg = _load_models()

# Optional save directories (disabled by default)
SCREEN_SAVE_DIRECTORY = REPO_DIR / "screen_images_end2end"
OBJ_SAVE_DIRECTORY = REPO_DIR / "obj_images_end2end"
SCREEN_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
OBJ_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _decode_b64_image(img_b64: str):
    img_bytes = base64.b64decode(img_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@app.websocket("/ws/process-gesture-context-stream")
async def websocket_endpoint(websocket: WebSocket):
    global latest_gesture_embed, latest_context_embed, is_left_hand_wrist_based
    await websocket.accept()
    print("Unity client connected (end2end).")

    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)

            # ---- Gesture data ----
            left_hand_data = data.get("left_hand")
            right_hand_data = data.get("right_hand")

            gesture_result, is_left = gesture_processor.process(left_hand_data, right_hand_data)
            if gesture_result is not None:
                latest_gesture_embed = gesture_result.detach()
                is_left_hand_wrist_based = is_left
                print("Gesture embedding shape:", latest_gesture_embed.shape)

            # ---- Context data ----
            screen_capture = data.get("screen_capture")
            object_captures = data.get("object_captures")
            available_tools = data.get("available_tools") or []
            inventory_labels = []

            # Build inventory labels from available_tools (list of item classes; id or name)
            for tool in available_tools:
                if isinstance(tool, dict):
                    label = tool.get("id") or tool.get("name") or tool.get("label")
                else:
                    label = str(tool)
                if label:
                    inventory_labels.append(str(label))
            if inventory_labels:
                print(f"Available tools: {inventory_labels}")

            if screen_capture:
                try:
                    img_cv = _decode_b64_image(screen_capture)
                    if img_cv is not None:
                        print(f"Decoded screen frame, shape: {img_cv.shape}")

                        # Save (optional)
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"screen_{timestamp}.jpg"
                        file_path = SCREEN_SAVE_DIRECTORY / filename
                        cv2.imwrite(str(file_path), img_cv)

                        # Convert to PIL and update buffer
                        pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                        frame_buffer.append(pil_img)

                        if len(frame_buffer) > 0:
                            with torch.no_grad():
                                context_embedding = context_model.forward_frames(list(frame_buffer))
                            latest_context_embed = context_embedding.detach()
                            print("Context embedding shape:", latest_context_embed.shape)
                except Exception as e:
                    print(f"Error decoding screen image: {e}")

            if object_captures:
                print(f"--- Received {len(object_captures)} captured objects (saved only) ---")
                for capture in object_captures:
                    label = capture.get("label", "unknown_object") if isinstance(capture, dict) else "unknown_object"
                    img_b64 = capture.get("image_base64") if isinstance(capture, dict) else None
                    if not img_b64:
                        continue
                    try:
                        img_cv = _decode_b64_image(img_b64)
                        if img_cv is not None:
                            print(f"  > Decoded object '{label}', shape: {img_cv.shape}")
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            filename = f"{label}_{timestamp}.jpg"
                            file_path = OBJ_SAVE_DIRECTORY / filename
                            cv2.imwrite(str(file_path), img_cv)
                    except Exception as e:
                        print(f"Error decoding object image for '{label}': {e}")

            # ---- Prediction ----
            predicted_tool = ""
            if latest_gesture_embed is not None and latest_context_embed is not None and inventory_labels:
                g_in = latest_gesture_embed
                c_in = latest_context_embed
                if g_in.dim() == 1:
                    g_in = g_in.unsqueeze(0)
                if c_in.dim() == 1:
                    c_in = c_in.unsqueeze(0)
                with torch.no_grad():
                    pred_idx, item_logits = classifier.predict(
                        g_in,
                        c_in,
                        inventory_texts=[inventory_labels],
                    )
                idx = int(pred_idx.item())
                if 0 <= idx < len(inventory_labels):
                    predicted_tool = inventory_labels[idx]
                scores = item_logits[0, : len(inventory_labels)].detach().cpu().tolist()
                print(f"[Classifier] predicted tool: {predicted_tool} (idx={idx}) scores={scores}")

            response = {
                "predicted_tool": predicted_tool,
                "is_left_hand_wrist_based": is_left_hand_wrist_based,
            }
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("Unity client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")


# run server:
# uvicorn unity_server_end2end:app --host 0.0.0.0 --port 8000

