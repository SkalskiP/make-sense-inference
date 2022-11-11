from dataclasses import dataclass
from typing import List, Optional, Callable, Any

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin


@dataclass(frozen=True)
class BoundingBox(DataClassJsonMixin):
    left_top_x: float
    left_top_y: float
    right_bottom_x: float
    right_bottom_y: float


@dataclass(frozen=True)
class PredictionDetails(DataClassJsonMixin):
    confidence: float
    class_id: int
    class_name: Optional[str] = None


@dataclass(frozen=True)
class Detection(DataClassJsonMixin):
    bounding_box: BoundingBox
    prediction_details: PredictionDetails


@dataclass(frozen=True)
class InferenceResult(DataClassJsonMixin):
    image_height: int
    image_width: int
    detections: List[Detection]


InferenceResultDict = dict

ModelObject = Any  # this can be anything even, tensorflow if someone wanted
InferenceFunction = Callable[
    [ModelObject, List[np.ndarray], torch.device], List[InferenceResult]
]


@dataclass(frozen=True)
class ModelConfig(DataClassJsonMixin):
    model_family: str
    weights_file_name: Optional[str]
    factory_parameters: dict
