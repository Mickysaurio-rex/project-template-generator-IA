#import datetime
from typing import Any
import numpy as np
from ultralytics import YOLO
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
OBJECT_DETECTION_PATH = "ssd_mobilenet_v2.tflite"


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
    
    # def get_predicction_report(self):
    #     for sec in self.detection_dict.items():
    #         report = {
    #             "Nombre": sec.name[0],
    #             "Probabilidad": sec.score[0],
    #             "Punto X": sec.bbox.origin_x,
    #             "Punto Y": sec.bbox.origin_y,
    #             "Alto": sec.bbox.width,
    #             "Ancho": sec.bbox.height,
    #             "Fecha y Hora": sec.date[0]
    #         }
        
    #     return report
        
