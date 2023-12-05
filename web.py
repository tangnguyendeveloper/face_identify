from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import uvicorn

from FaceIdentify_TFlite.facenet_tflite import MTCNNFaceNetTFlite
from FaceIdentify_TFlite.utils import LoadModelTFlite, TFlitePredict
from FaceIdentify_TFlite.model_path import identify_path
import numpy as np

from serial import Serial
from time import time as now
import logging


app = FastAPI(debug=True)
templates = Jinja2Templates(directory="./template")

capture = None
bbox = None
detector = None
Identify_model = None


font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.5
font_thickness = 1
font_color = (0, 255, 0)  # White color in BGR
position = (50, 200)  # (x, y) coordinates

zigbee = None


LABLE = ["Nhi", "Thuy", "Unknown user"]

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def capture_image():
    global capture
    return capture.read()

async def gen_frames():

    _open_door = False

    while not _open_door:
        try:
            ok, frame = await capture_image()
            if not ok:
                break
            
            frame, _open_door = await open_door(frame)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            if _open_door:    
                await DoorAction(_open_door)
            
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        except Exception as e:
            print(f"Error in gen_frames: {e}")
            break

@app.get("/serve", include_in_schema=False)
async def serve_video():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


async def ZigbeeReceiveWithTimeout(timeout_in_second: int):
    start = now()

    while now() - start < timeout_in_second :
        ack = zigbee.read(4).decode('utf-8')
        if ack == "OK!\n":
            return True
        
    return False

async def DoorAction(open_door: bool) -> bool:
    global zigbee
    if open_door:
        zigbee.write(b"open\n")
    else:
        zigbee.write(b"close\n")

    ok_ack = await ZigbeeReceiveWithTimeout(3)
    if ok_ack:
        logging.info("DoorAction OK")
    else:
        logging.warn("No ACK DoorAction")


async def open_door(img):
    global detector, Identify_model, bbox

    frame = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    results_detector = detector.extract(frame)
    text = ""
    _open_door = False

    if len(results_detector) != 1:
        text = "Please point the camera at your face \n (Refest page to try again)."
        cv2.putText(img, text, position, font, font_size, font_color, font_thickness)
        return img, _open_door

    bbox = results_detector[0]['box']
    x, y, w, h = bbox
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5) 

    x = np.asarray(results_detector[0]['embedding'], dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    y = TFlitePredict(Identify_model, x)
    y = np.round(y, 6)
    y = (y == 1.0).flatten()

    if y[0]:
        _open_door = True
        text = f"You are {LABLE[0]}, The door will be open in 5 seconds, (Refest page to try again)."
    elif y[1]:
        _open_door = True
        text = f"You are {LABLE[1]}, The door will be open in 5 seconds, (Refest page to try again)."
    else:
        text = f"You are {LABLE[2]}, can not open the door, (Refest page to try again)."
    
    cv2.putText(img, text, position, font, font_size, font_color, font_thickness)
    return img, _open_door


@app.on_event("startup")
async def startup_event():
    global capture, detector, Identify_model, zigbee

    zigbee = Serial(
        port="/dev/ttyUSB0", baudrate=9600, timeout=1,
    )

    capture = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    detector = MTCNNFaceNetTFlite()
    Identify_model = LoadModelTFlite(identify_path)

    await DoorAction(False)


@app.on_event("shutdown")
async def shutdown_event():
    global capture, zigbee
    
    if capture.isOpened():
        capture.release()
    if zigbee.is_open():
        zigbee.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80, log_level="info")
