import json
import cv2
import requests

from log.logger import logger

# tensorflow
import tensorflow as tf

# fast api
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# model
from pydantic import BaseModel

from image_processing import bbox_converter

# rich
from rich.console import Console

console = Console()

console.print('#################', style='bold cyan')
console.print('#YOLOv4 settings#', style='bold cyan')
console.print('#################', style='bold cyan')
# image size
size = 1024
console.print("[bold cyan]Image Size:[/bold cyan] ", size)
# iou threshold
iou = 0.45
console.print("[bold cyan]Iou:[/bold cyan] ", iou)
# score threshold
score = 0.25
console.print("[bold cyan]threshold score:[/bold cyan] ", score)

console.print('########################', style='bold cyan')
console.print('#Tensorflow Serving URL#', style='bold cyan')
console.print('########################', style='bold cyan')
MODEL_URI = 'http://localhost:8501/v1/models/yolov4:predict'
console.print("[bold cyan]MODEL_URI:[/bold cyan] ", MODEL_URI)

tags = [
    {
        'name': 'comm',
        'description': 'check server alive',
    },
    {
        'name': 'image_path',
        'description': 'read image',
    }
]

class ImagePath(BaseModel):
    image_path: str

app = FastAPI(
    title="Yolov4 Detection(TF Serving)",
    version="v1",
    description="detection",
    openapi_tags=tags,
)

@app.on_event("startup")
def startup():
    logger.info('Application start')

@app.on_event("shutdown")
def shutdown():
    logger.info('Application shutdown')

@RequestValidationError
@app.post('api/detection/', tags=['image_path'], summary="1.Trash 2.Car 3.Building 4.Ocean 5.Land")
def image_detection(data:ImagePath):
    json_data = jsonable_encoder(data)
    image_path = json_data['image_path']
    logger.info(image_path)
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # inferencing
    logger.info('detection start')
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, (size, size))
    image = image / 255
    img_np = image.numpy()

    predict_request = json.dumps({
        'instances': img_np.tolist()
    })

    response = requests.post(MODEL_URI, data=predict_request)
    json_response = json.loads(response.text)

    for key, value in json_response.items():
        value = tf.convert_to_tensor(value)
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )

    pred_bbox = [boxes.numpy(), classes.numpy(), valid_detections.numpy()]
    # convert coordinates
    object_coords = bbox_converter.bbox_convert(original_image, pred_bbox)

@app.head('/api/alive', tags=['comm'], responses={
    204: {"description": 'Alive Server'},
    404: {"description": 'Dead Server'}
}, status_code=204, )
def server_check():
    return JSONResponse(status_code=204)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
