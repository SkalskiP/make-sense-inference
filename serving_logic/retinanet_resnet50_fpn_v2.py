from functools import partial
from typing import Optional, Tuple, List, Callable

import numpy as np
import torch
from torch import nn
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

from entities import (
    ModelObject,
    InferenceFunction,
    InferenceResult,
    BoundingBox,
    PredictionDetails,
    Detection,
)

PREDICTION_BBOXES_KEY = "boxes"
PREDICTION_SCORES_KEY = "scores"
PREDICTION_LABELS_KEY = "labels"


def load_retinanet_model(
    device: torch.device,
    weights_path: Optional[str] = None,
) -> Tuple[ModelObject, InferenceFunction]:
    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    model = retinanet_resnet50_fpn_v2(weights=weights).to(device)
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model, partial(
        infer, transforms=weights.transforms(), class_names=weights.meta["categories"]
    )


def infer(
    model: nn.Module,
    images: List[np.ndarray],
    _: torch.device,
    transforms: Callable[[torch.Tensor], torch.Tensor],
    class_names: List[str],
) -> List[InferenceResult]:
    pre_processed_images = pre_process_object_detector_input(
        images=images, model_transforms=transforms
    )
    with torch.no_grad():
        inference_results = model(pre_processed_images)
    return post_process_retina_net_predictions(
        images=images, raw_predictions=inference_results, class_names=class_names
    )


def pre_process_object_detector_input(
    images: list[np.ndarray], model_transforms: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    raw_images = [torch.from_numpy(image).permute(2, 0, 1) for image in images]
    return torch.stack([model_transforms(image) for image in raw_images], dim=0)


def post_process_retina_net_predictions(
    images: List[np.ndarray],
    raw_predictions: List[dict],
    class_names: List[str],
) -> List[InferenceResult]:
    return [
        post_process_retina_net_prediction(
            image=image, prediction=prediction, class_names=class_names
        )
        for image, prediction in zip(images, raw_predictions)
    ]


def post_process_retina_net_prediction(
    image: np.ndarray, prediction: dict, class_names: List[str]
) -> InferenceResult:
    image_height, image_width = image.shape[:2]
    iterable = zip(
        prediction[PREDICTION_BBOXES_KEY],
        prediction[PREDICTION_SCORES_KEY],
        prediction[PREDICTION_LABELS_KEY],
    )
    detections = []
    for raw_bbox, score, class_id in iterable:
        detection_bounding_box = create_bounding_box(image=image, raw_bbox=raw_bbox)
        prediction_details = prepare_prediction_details(
            score=score, class_id=class_id, class_names=class_names
        )
        detection = Detection(
            bounding_box=detection_bounding_box, prediction_details=prediction_details
        )
        detections.append(detection)
    return InferenceResult(
        image_height=image_height,
        image_width=image_width,
        detections=detections,
    )


def create_bounding_box(image: np.ndarray, raw_bbox: torch.Tensor) -> BoundingBox:
    image_height, image_width = image.shape[:2]
    left_top_x = raw_bbox[0].item() / image_width
    left_top_y = raw_bbox[1].item() / image_height
    right_bottom_x = raw_bbox[2].item() / image_width
    right_bottom_y = raw_bbox[3].item() / image_height
    return BoundingBox(
        left_top_x=left_top_x,
        left_top_y=left_top_y,
        right_bottom_x=right_bottom_x,
        right_bottom_y=right_bottom_y,
    )


def prepare_prediction_details(
    score: torch.Tensor, class_id: torch.Tensor, class_names: List[str]
) -> PredictionDetails:
    return PredictionDetails(
        confidence=score.item(),
        class_id=class_id.item(),
        class_name=class_names[class_id],
    )
