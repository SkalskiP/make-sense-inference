from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import torch
from ts.context import Context

from constants import REQUEST_IMAGE_FIELD
from entities import InferenceFunction, ModelObject, InferenceResultDict
from serving_utils import extract_device_from_context, load_model


class Handler:
    def __init__(self):
        self.__model: Optional[ModelObject] = None
        self.__inference_function: Optional[InferenceFunction] = None
        self.__device: Optional[torch.device] = None

    def initialize(self, context: Context) -> None:
        self.__device = extract_device_from_context(context=context)
        self.__model, self.__inference_function = load_model(
            context=context, device=self.__device
        )

    def handle(
        self, requests_body: List[Dict[str, Any]], context: Context
    ) -> List[InferenceResultDict]:
        self.__ensure_handler_initialised()
        self.__ensure_batch_size_equals_one(requests_body=requests_body)
        request_images = [
            decode_request(request_body=request_body) for request_body in requests_body
        ]
        inference_results = self.__inference_function(  # type: ignore
            self.__model, request_images, self.__device
        )
        return [inference_result.to_dict() for inference_result in inference_results]

    def __ensure_handler_initialised(self) -> None:
        to_verify_not_empty = [self.__device, self.__model, self.__inference_function]
        if any(e is None for e in to_verify_not_empty):
            raise RuntimeError("Handler not initialised correctly.")

    def __ensure_batch_size_equals_one(
        self, requests_body: List[Dict[str, Any]]
    ) -> None:
        if len(requests_body) > 1:
            raise NotImplementedError(
                "To fully correctly handle torchserve requests batching one must properly dispatch "
                "potential errors for specific responses - such that one faulty input does not destroy "
                "whole batch of inference by causing Exception. This is however something not tackled in "
                "official torchserve handlers (they also handle only batch_size=1 to my knowledge) - at the "
                "same time doable easily (just original author of this code is to lazy). To support that in the "
                "future - InferenceFunction accepts list of decoded images and returns list of inference results."
            )


def decode_request(request_body: Dict[str, Any]) -> np.ndarray:
    raw_image = request_body[REQUEST_IMAGE_FIELD]
    bytes_array = np.asarray(raw_image, dtype=np.uint8)
    return cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)[:, :, ::-1]
