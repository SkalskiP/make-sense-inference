import json
import os.path
from typing import Tuple, Union

import torch
from ts.context import Context

from constants import MODEL_CONFIG_FILE_NAME, YOLO_V5_FACTORY_NAME
from entities import ModelObject, InferenceFunction, ModelConfig


def extract_device_from_context(context: Context) -> torch.device:
    return torch.device(
        "cuda:" + str(context.system_properties.get("gpu_id"))
        if torch.cuda.is_available()
        else "cpu"
    )


def load_model(
    context: Context, device: torch.device
) -> Tuple[ModelObject, InferenceFunction]:
    model_config = load_model_config(context=context)
    # this will be ugly, but if we want to keep specific model dependencies separated
    # which is quite reasonable otherwise we may not be able to resolve environment if we add
    # multiple contradicting libraries later on (we need local import such that specific
    # dependencies are only imported for specific models). That itself may cause problems if
    # dependent model packages imports their local modules of the same name...
    if model_config.model_factory == YOLO_V5_FACTORY_NAME:
        from yolov5 import load_yolov5_model, YoloV5FactoryParameters

        yolo_parameters = YoloV5FactoryParameters.from_dict(
            model_config.factory_parameters
        )
        weights_path = (
            os.path.join(
                context.system_properties["model_dir"], model_config.weights_file_name
            )
            if model_config.weights_file_name is not None
            else None
        )
        return load_yolov5_model(
            factory_parameters=yolo_parameters, device=device, weights_path=weights_path
        )
    raise NotImplementedError(
        f"Model factory `{model_config.model_factory}` not implemented."
    )


def load_model_config(context: Context) -> ModelConfig:
    model_config_path = os.path.join(
        context.system_properties["model_dir"], MODEL_CONFIG_FILE_NAME
    )
    model_config_raw = load_json_file(path=model_config_path)
    if model_config_raw is None:
        raise RuntimeError("Malformed (empty) config found.")
    return ModelConfig.from_dict(model_config_raw)


def load_json_file(path: str) -> Union[dict, list, None]:
    with open(path) as f:
        return json.load(f)