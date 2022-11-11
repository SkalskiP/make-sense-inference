from dataclasses import dataclass
from functools import partial
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from dataclasses_json import DataClassJsonMixin
from torch import nn

from entities import (
    ModelObject,
    InferenceFunction,
    InferenceResult,
    BoundingBox,
    PredictionDetails,
    Detection,
)

MODEL_REPOSITORY = "ultralytics/yolov5"


@dataclass(frozen=True)
class YoloV5FactoryParameters(DataClassJsonMixin):
    model_version: str
    inference_size: int


def load_yolov5_model(
    factory_parameters: YoloV5FactoryParameters,
    device: torch.device,
    weights_path: Optional[str] = None,
) -> Tuple[ModelObject, InferenceFunction]:
    model = torch.hub.load(
        MODEL_REPOSITORY, factory_parameters.model_version, device=device
    )
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model, partial(infer, inference_size=factory_parameters.inference_size)


def infer(
    model: nn.Module, images: List[np.ndarray], _: torch.device, inference_size: int
) -> List[InferenceResult]:
    with torch.no_grad():
        inference_results = model(images, size=inference_size)
    return [
        prepare_inference_result(image=image, result=result)
        for image, result in zip(images, inference_results.pandas().xyxy)
    ]


def prepare_inference_result(
    image: np.ndarray, result: pd.DataFrame
) -> InferenceResult:
    image_height, image_width = image.shape[:2]
    detections = []
    for _, detection in result.iterrows():
        detection_bounding_box = prepare_detection_bounding_box(
            detection=detection,
            image_height=image_height,
            image_width=image_width,
        )
        prediction_details = prepare_prediction_details(detection=detection)
        detections.append(
            Detection(
                bounding_box=detection_bounding_box,
                prediction_details=prediction_details,
            )
        )
    return InferenceResult(
        image_height=image_height,
        image_width=image_width,
        detections=detections,
    )


def prepare_detection_bounding_box(
    detection: pd.Series, image_height: int, image_width: int
) -> BoundingBox:
    left_top_x = detection.xmin / image_width
    left_top_y = detection.ymin / image_height
    right_bottom_x = detection.xmax / image_width
    right_bottom_y = detection.ymax / image_height
    return BoundingBox(
        left_top_x=left_top_x,
        left_top_y=left_top_y,
        right_bottom_x=right_bottom_x,
        right_bottom_y=right_bottom_y,
    )


def prepare_prediction_details(detection: pd.Series) -> PredictionDetails:
    return PredictionDetails(
        confidence=detection.confidence,
        class_id=detection["class"],
        class_name=detection["name"],
    )
