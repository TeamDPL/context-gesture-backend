from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import base64  # base64 decoding
import numpy as np  # image array
import cv2 # image processing
from pathlib import Path
import datetime
from collections import deque
from PIL import Image
import torch
from owl_sgvit_gru import OWLSGVitConfig, OWLSGVitGRU
from gesture_encoder import GestureStreamProcessor

# ==== Context encoder hyperparams ====
TEXT_QUERIES     = ["object"]  
DET_THRESHOLD    = 0.2
MAX_OBJECTS      = 64
TOPK_RELATIONS   = 32
FRAME_DIM        = 768
USE_ATTN_POOL    = True
FRAME_BUFFER_LEN = 10 
# =====================================

# model config
cfg = OWLSGVitConfig(
    text_queries=TEXT_QUERIES,
    det_threshold=DET_THRESHOLD,
    max_objects=MAX_OBJECTS,
    topk_relations=TOPK_RELATIONS,
    frame_dim=FRAME_DIM,
    use_attn_pool=USE_ATTN_POOL,
)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
context_model = OWLSGVitGRU(cfg, device=device)

# the number of frame: 10
frame_buffer = deque(maxlen=FRAME_BUFFER_LEN)

# Gesture Encoder Initialization
gesture_processor = GestureStreamProcessor(device=device)

app = FastAPI()

SCREEN_SAVE_DIRECTORY = Path("screen_images")
SCREEN_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

OBJ_SAVE_DIRECTORY = Path("obj_images")
OBJ_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

@app.websocket("/ws/process-gesture-context-stream")
async def websocket_endpoint(websocket: WebSocket):
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
            gesture_feature = gesture_processor.process(left_hand_data, right_hand_data)
            
            if gesture_feature is not None:
                print("Gesture Feature shape:", gesture_feature.shape)
                print("Gesture Feature (first 5 rows):", gesture_feature.head(5))

            if left_hand_data:
                print("L Hand World X:", left_hand_data)

            if right_hand_data:
                print("R Hand World X:", right_hand_data)

            # screen data
            screen_capture = data.get("screen_capture")
            object_captures = data.get("object_captures")

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
                                c = context_model.forward_frames(list(frame_buffer))
                            print("context embedding from last 10 frames:", c.shape)    # Print only yet

                except Exception as e:
                    print(f"Error decoding image: {e}")
            
            if object_captures:
                print(f"--- Processing frame with {len(object_captures)} captured objects ---")
                
                for capture in object_captures:
                    label = capture.get("label", "unknown_object")
                    img_b64 = capture.get("image_base64")
                    
                    if not img_b64:
                        continue
                        
                    try:
                        # Decode each image
                        img_bytes = base64.b64decode(img_b64)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if img_cv is not None:
                            print(f"  > Decoded image for object: '{label}', shape: {img_cv.shape}")
                            
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            filename = f"{label}_{datetime.datetime.now().strftime('%f')}.jpg"
                            
                            # 2. Create the full file path
                            file_path = OBJ_SAVE_DIRECTORY / filename

                            # 3. Save the image to disk
                            cv2.imwrite(str(file_path), img_cv)
                            
                            print(f"Successfully saved image to {file_path}")

                    except Exception as e:
                        print(f"Error decoding image for '{label}': {e}")
            
            result = {"action": "Equip", "tool": "Axe_From_WS"}
            
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print("Unity client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")

# run server:
# uvicorn unity_server:app --host 0.0.0.0 --port 8000