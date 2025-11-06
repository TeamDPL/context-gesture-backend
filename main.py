from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json

app = FastAPI()

@app.websocket("/ws/process-gesture-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Unity client connected.")
    
    try:
        while True:
            data_str = await websocket.receive_text()
            
            data = json.loads(data_str)
            
            left_hand_data = data.get("left_hand")
            right_hand_data = data.get("right_hand")
            
            if left_hand_data:
                print("L Hand World X:", left_hand_data)

            if right_hand_data:
                print("R Hand World X:", right_hand_data)
            
            result = {"action": "Equip", "tool": "Axe_From_WS"}
            
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print("Unity client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")

# run server:
# uvicorn main:app --host 0.0.0.0 --port 8000