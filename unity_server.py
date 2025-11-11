from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import base64  # base64 decoding
import numpy as np  # image array
import cv2 # image processing
from pathlib import Path
import datetime

app = FastAPI()

SAVE_DIRECTORY = Path("unity_context_images")
SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

@app.websocket("/ws/process-gesture-stream")
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
            screen_capture_b64 = data.get("screen_capture")
            
            if screen_capture_b64:
                try:
                    img_bytes = base64.b64decode(screen_capture_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img_cv is not None:
                        print(f"Successfully decoded image, shape: {img_cv.shape}")

                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"capture_{timestamp}.jpg"
                        
                        # 2. Create the full file path
                        file_path = SAVE_DIRECTORY / filename

                        # 3. Save the image to disk
                        #    cv2.imwrite() takes the (string) path and the image object
                        cv2.imwrite(str(file_path), img_cv)
                        
                        print(f"Successfully saved image to {file_path}")
                        
                    # SLAM / CV PROCESSING BELOW
                    
                except Exception as e:
                    print(f"Error decoding image: {e}")
            
            result = {"action": "Equip", "tool": "Axe_From_WS"}
            
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print("Unity client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")

# run server:
# uvicorn unity_server:app --host 0.0.0.0 --port 8000