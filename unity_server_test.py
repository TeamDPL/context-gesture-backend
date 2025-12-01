from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import base64  # base64 decoding
import numpy as np  # image array
import cv2 # image processing
from pathlib import Path
import datetime

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

                        # cv2.imwrite(str(file_path), img_cv)
                        # print(f"Successfully saved image to {file_path}")
                        
                    # SLAM / CV PROCESSING BELOW
                    
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
                            # cv2.imwrite(str(file_path), img_cv)
                            # print(f"Successfully saved image to {file_path}")

                    except Exception as e:
                        print(f"Error decoding image for '{label}': {e}")
            
            result = {"ID": "CuttingBoard"}
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print("Unity client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")

# run server:
# uvicorn unity_server_test:app --host 0.0.0.0 --port 8000