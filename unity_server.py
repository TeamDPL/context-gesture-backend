import os
import json
import base64  # base64 decoding
import datetime
from collections import deque
from pathlib import Path

import cv2  # image processing
import numpy as np  # image array
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from gesture_context_classifier import FusionTreeConfig, GestureContextTreeClassifier
from gesture_encoder import GestureStreamProcessor
from owl_sgvit_gru import OWLSGVitConfig, OWLSGVitGRU

# ==== Context encoder hyperparams ====
TEXT_QUERIES     = ["object"]  
DET_THRESHOLD    = 0.2
MAX_OBJECTS      = 64
TOPK_RELATIONS   = 32
FRAME_DIM        = 768
USE_ATTN_POOL    = True
FRAME_BUFFER_LEN = 10 
# =====================================

GESTURE_DIM = 512  # output size of GestureStreamProcessor encoder (TD-GCN dual hands)

# Local HF model paths (set env vars to override defaults)
CLIP_LOCAL_PATH = os.getenv("CLIP_MODEL_PATH", "./hf_models/clip-vit-base-patch32")
CLIP_LOCAL_FILES_ONLY = os.getenv("CLIP_LOCAL_FILES_ONLY", "true").lower() in ("1", "true", "yes")
OWL_MODEL_PATH = os.getenv("OWL_VIT_MODEL_PATH", "./hf_models/owlvit-base-patch16")
OWL_PROCESSOR_PATH = os.getenv("OWL_VIT_PROCESSOR_PATH", "./hf_models/owlvit-base-patch16")
OWL_LOCAL_FILES_ONLY = os.getenv("OWL_VIT_LOCAL_FILES_ONLY", "true").lower() in ("1", "true", "yes")

# model config
cfg = OWLSGVitConfig(
    text_queries=TEXT_QUERIES,
    det_threshold=DET_THRESHOLD,
    max_objects=MAX_OBJECTS,
    topk_relations=TOPK_RELATIONS,
    frame_dim=FRAME_DIM,
    use_attn_pool=USE_ATTN_POOL,
    owl_pretrained_path=OWL_MODEL_PATH,
    processor_pretrained_path=OWL_PROCESSOR_PATH,
    local_files_only=OWL_LOCAL_FILES_ONLY,
)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
context_model = OWLSGVitGRU(cfg, device=device)

# the number of frame: 10
frame_buffer = deque(maxlen=FRAME_BUFFER_LEN)

# Gesture Encoder Initialization
gesture_processor = GestureStreamProcessor(device=device)

# Gesture-Context Classifier (inventory-driven)
classifier_cfg = FusionTreeConfig(
    gesture_dim=GESTURE_DIM,
    context_dim=FRAME_DIM,
    fusion_dim=256,
    tree_depth=3,
    use_clip_inventory=True,
    clip_model_name="openai/clip-vit-base-patch32",
    clip_local_path=CLIP_LOCAL_PATH,
    clip_local_files_only=CLIP_LOCAL_FILES_ONLY,
    use_leaf_queries=True,
    leaf_text_prompts=[
        "combat situation, choose weapon",
        "cooking situation, choose kitchen tool",
        "crafting situation, choose building tool",
    ],
)
classifier = GestureContextTreeClassifier(classifier_cfg, device=device)

# runtime caches
latest_gesture_embed = None  # torch.Tensor [1, G]
latest_context_embed = None  # torch.Tensor [1, C]

app = FastAPI()

SCREEN_SAVE_DIRECTORY = Path("screen_images")
SCREEN_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

OBJ_SAVE_DIRECTORY = Path("obj_images")
OBJ_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

@app.websocket("/ws/process-gesture-context-stream")
async def websocket_endpoint(websocket: WebSocket):
    global latest_gesture_embed, latest_context_embed
    await websocket.accept()
    print("Unity client connected.")
    
    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            
            # gesture data
            left_hand_data = data.get("left_hand")
            right_hand_data = data.get("right_hand")

            # Process Hand Data
            gesture_embedding = gesture_processor.process(left_hand_data, right_hand_data)
            if gesture_embedding is not None:
                latest_gesture_embed = gesture_embedding.detach()
                print("Gesture embedding shape:", gesture_embedding.shape)

            if left_hand_data:
                print("L Hand World X:", left_hand_data)

            if right_hand_data:
                print("R Hand World X:", right_hand_data)

            # screen data
            screen_capture = data.get("screen_capture")
            object_captures = data.get("object_captures")
            available_tools = data.get("available_tools") or []
            inventory_labels = []

            # Build inventory labels from available_tools (list of item classes; id is the name)
            if available_tools:
                for tool in available_tools:
                    if isinstance(tool, dict):
                        label = tool.get("id") or tool.get("name") or tool.get("label")
                    else:
                        label = str(tool)
                    if label:
                        inventory_labels.append(str(label))
                print(f"Available tools: {inventory_labels}")

            if screen_capture:
                try:
                    img_bytes = base64.b64decode(screen_capture)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img_cv is not None:
                        print(f"Successfully decoded image, shape: {img_cv.shape}")

                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"screen_{timestamp}.jpg"
                        
                        file_path = SCREEN_SAVE_DIRECTORY / filename

                        cv2.imwrite(str(file_path), img_cv)
                        
                        print(f"Successfully saved image to {file_path}")
                        
                        # CONTEXT-ENCODER RECALL BELOW
                        pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                        frame_buffer.append(pil_img)

                        if len(frame_buffer) == frame_buffer.maxlen:
                            with torch.no_grad():
                                context_embedding = context_model.forward_frames(list(frame_buffer))
                            latest_context_embed = context_embedding.detach()
                            print("context embedding from last 10 frames:", context_embedding.shape)    # Print only yet

                except Exception as e:
                    print(f"Error decoding image: {e}")
            
            if object_captures:
                # Optional: keep saving images for later use, but do NOT use as inventory.
                print(f"--- Received {len(object_captures)} captured objects (not used for inventory) ---")
                
                for capture in object_captures:
                    label = capture.get("label", "unknown_object") if isinstance(capture, dict) else "unknown_object"
                    img_b64 = capture.get("image_base64") if isinstance(capture, dict) else None
                    
                    if not img_b64:
                        continue
                        
                    try:
                        img_bytes = base64.b64decode(img_b64)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if img_cv is not None:
                            print(f"  > Decoded image for object: '{label}', shape: {img_cv.shape}")
                            
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            filename = f"{label}_{datetime.datetime.now().strftime('%f')}.jpg"
                            
                            file_path = OBJ_SAVE_DIRECTORY / filename
                            cv2.imwrite(str(file_path), img_cv)
                            
                            print(f"Successfully saved image to {file_path}")

                    except Exception as e:
                        print(f"Error decoding image for '{label}': {e}")
            
            # Run classifier when all signals are ready
            predicted_tool = None
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

            # Unity expects the predicted tool id/name as a raw string (empty if none)
            result = predicted_tool if predicted_tool is not None else ""
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print("Unity client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")

# run server:
# uvicorn unity_server:app --host 0.0.0.0 --port 8000
