from starlette.middleware.cors import CORSMiddleware
from src.llm_service import TemplateLLM
from src.prompts import ProjectParams
from src.parsers import ProjectIdeas
from src.config import get_settings
import io
import cv2
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    status,
    Depends,
)
from fastapi.responses import Response, FileResponse
import numpy as np
from PIL import Image, UnidentifiedImageError
import mediapipe as mp
from typing import Any
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SETTINGS = get_settings()
OBJECT_DETECTION_PATH = "src/ssd_mobilenet_v2.tflite"


PERSON_COLOR = (70,223,49)
BOOK_COLOR = (20,37,241)
BED_COLOR = (9,234,200)
COUCH_COLOR = (252, 233, 1)
CHAIR_COLOR = (166, 29, 245)
CAR_COLOR = (246, 18, 187)



app = FastAPI(
    title=SETTINGS.service_name,
    version=SETTINGS.k_revision
)

class ObjectDetector:
    def __init__(self, model_path=OBJECT_DETECTION_PATH):
        base_options = python.BaseOptions(model_asset_path=OBJECT_DETECTION_PATH)
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.25,
                                       )
        self.model = vision.ObjectDetector.create_from_options(options)
        self.detection_dict = {}
    
    def get_self_detection_dict(self):
        return self.detection_dict

    def predict_image(self, image_array: np.ndarray):
        mp_image= mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
        detection = self.model.detect(mp_image)
        results = []
        for detection in detection.detections:
            bbox = detection.bounding_box
            categories = detection.categories
            detection_dict = {
                "bbox": [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height],
                "name": [(nm.category_name) for nm in categories],
                "score": [(sc.score) for sc in categories],
                #"date": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

            }
            results.append(detection_dict)

        self.detection_dict = detection_dict
        return results


object_predictor = ObjectDetector()
list_predict = []
def get_object_detector():
    return object_predictor

def predict_uploadfile(predictor, file):
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="Not an image"
        )
    img_obj = Image.open(img_stream)
    img_array = np.array(img_obj)
    return predictor.predict_image(img_array), img_array



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_llm_service():
    return TemplateLLM()


@app.post("/generate")
def generate_project(params: ProjectParams, service: TemplateLLM = Depends(get_llm_service)) -> ProjectIdeas:
    return service.generate(params)

@app.post("/predecir_y_anotar_objetos", responses={
    200: {"content": {"image/jpeg": {}}}
    })
def detect_objects(
    file: UploadFile = File(...), 
    predictor: ObjectDetector = Depends(get_object_detector)
) -> Response:
    results, img = predict_uploadfile(predictor, file)
    color_adapt = (0,0,0)
    val_dis = 80
    object_count = {}
    for result in results:
        bbox = result['bbox']
        name = result['name']
        score = result['score']
        #date = result['date']

        dict ={
            "nombre": str(name[0]),
            "probabilidad": str(score[0]),
            "punto_x": str(bbox[0]),
            "punto_y": str(bbox[1]),
            "alto": str(bbox[2]),
            "ancho": str(bbox[3]),
            #"fecha": str(date[0]) 
        }
        list_predict.append(dict)

        if name[0] not in object_count:
            object_count[name[0]] =  1
        else : 
            object_count[name[0]] = object_count[name[0]] + 1
        
        if name[0] == "book":
            color_adapt = BOOK_COLOR
        elif name[0] == "bed":
            color_adapt = BED_COLOR
        elif name[0] == "person":
            color_adapt = PERSON_COLOR
        elif name[0] == "chair":
            color_adapt = CHAIR_COLOR
        elif name[0] == "car":
            color_adapt = CAR_COLOR
        elif name[0] == "couch":
            color_adapt = COUCH_COLOR
        else:
            color_adapt = (0,0,0)


        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                            color_adapt, 2)
        cv2.putText(
            img, 
            name[0], 
            (bbox[0], bbox[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_adapt,
            2,
        )
        
        

    for key, val in object_count.items():
        cv2.putText(
            img, 
            f"{key}: {val}", 
            (20, val_dis), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2,
        )
        val_dis = val_dis + 45

    
    img_pil = Image.fromarray(img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.get("/")
def root():
    return {"status": "OK"}
