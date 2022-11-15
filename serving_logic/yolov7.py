import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Tuple, List, Optional

import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import resize
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


INFERENCE_SCALING = 255


@dataclass(frozen=True)
class YoloV7FactoryParameters(DataClassJsonMixin):
    inference_size: Tuple[int, int]
    classes: List[str]
    confidence_threshold: float
    iou_threshold: float


def load_yolov7_model(
    factory_parameters: YoloV7FactoryParameters,
    device: torch.device,
    weights_path: Optional[str] = None,
) -> Tuple[ModelObject, InferenceFunction]:
    if weights_path is None:
        raise RuntimeError("No model path provided")
    model = torch.jit.load(weights_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model, partial(infer, factory_parameters=factory_parameters)


def infer(
    model: nn.Module,
    images: List[np.ndarray],
    device: torch.device,
    factory_parameters: YoloV7FactoryParameters,
) -> List[InferenceResult]:
    images_batch = torch.stack(
        [torch.Tensor(image).permute(2, 0, 1) for image in images]
    )
    resized_batch = resize(
        images_batch, size=list(factory_parameters.inference_size)
    ).to(device)
    with torch.no_grad():
        predictions_batch = model(resized_batch / INFERENCE_SCALING)
        nms_results = non_max_suppression(
            predictions_batch[0],
            conf_thres=factory_parameters.confidence_threshold,
            iou_thres=factory_parameters.iou_threshold,
        )
    inference_results = []
    for original_image, resized_image, nms_result in zip(
        images_batch, resized_batch, nms_results
    ):
        scaled_predictions = scale_coords(
            resized_image.shape, nms_result.to("cpu"), original_image.shape
        )
        inference_result = prepare_inference_result(
            original_image=original_image,
            scaled_predictions=scaled_predictions,
            classes=factory_parameters.classes,
        )
        inference_results.append(inference_result)
    return inference_results


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
):
    """
    Code taken from: https://github.com/WongKinYiu/yolov7
    Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[
                :, 4:5
            ]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            logging.warning(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded
    return output


def xywh2xyxy(x):
    # Code taken from: https://github.com/WongKinYiu/yolov7
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Code taken from: https://github.com/WongKinYiu/yolov7
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def scale_coords(
    resized_shape: Tuple[int, int, int],
    coords: torch.Tensor,
    original_shape: Tuple[int, int, int],
) -> torch.Tensor:
    scale_h, scale_w = (
        original_shape[1] / resized_shape[1],
        original_shape[2] / resized_shape[2],
    )
    coords = (coords * torch.Tensor([scale_w, scale_h, scale_w, scale_h, 1, 1])).round()
    clip_coords(boxes=coords, img_shape=original_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Code taken from: https://github.com/WongKinYiu/yolov7
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def prepare_inference_result(
    original_image: torch.Tensor,
    scaled_predictions: torch.Tensor,
    classes: List[str],
) -> InferenceResult:
    _, image_height, image_width = original_image.shape
    detections = []
    for prediction in scaled_predictions:
        x1, y1, x2, y2 = prediction[:4].tolist()
        confidence = prediction[4].item()
        class_id = min(max(prediction[5].int().item(), 0), len(classes) - 1)
        bounding_box = BoundingBox(
            left_top_x=x1 / image_width,
            left_top_y=y1 / image_height,
            right_bottom_x=x2 / image_width,
            right_bottom_y=y2 / image_height,
        )
        prediction_details = PredictionDetails(
            confidence=confidence,
            class_id=class_id,
            class_name=classes[class_id],
        )
        detection = Detection(
            bounding_box=bounding_box,
            prediction_details=prediction_details,
        )
        detections.append(detection)
    return InferenceResult(
        image_height=image_height,
        image_width=image_width,
        detections=detections,
    )
