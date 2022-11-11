# makesense.ai inference
This repository is meant to be used as a template for serving models as part of integration with
[makesense.ai](https://github.com/SkalskiP/make-sense). Serving mechanism is based on TorchServe.


## Repository structure
* `docker` - directory with Dockerfile definitions
* `model_configurations` - directory to keep model configurations and weights
* `requirements` - directory with requirements definitions - in particular with dedicated requirements for
specific models
* `serving_config` - configuration of TorchServe server
* `serving_logic` - Pyton modules with code used for model inference - see further description in
documentation.

## How serving works?
Serving is based on TorchServe - and as a result user needs to prepare three components in order to deploy
their model. First of all - runtime environment with TorchServe must be prepared (and it is done
within docker images defined [here](./docker)). Second component inference handler - capable to load
model and handle requests (baseline one is prepared in [serving_logic](./serving_logic) package - their
internal structure will be described in next paragraph). And last, but not least - model and its config.
In our case - directory [model_configurations](./model_configurations) is meant to host model weights
and special configuration files used by `serving_logic` modules to load different models properly.

Having all three components one may build docker image and start receiving predictions. There are however few
concepts to be understood.

### TorchServe `*.mar` package
TorchServe accept special format of code and model bundles - `*.mar` - which help to decouple server from
models - in particular code used for model inference behind TorchServe, as well as model weights can be
prepared and injected into TorchServe environment. To make things simple at start - [docker definitions](./docker)
prepared in this repository are going to prepare model bundles automagically - based on build arg passed
into `docker build` command.

### How specific models are bundled and exposed in serving?
After closer look at repository structure - one may find the following pattern:
```
.
├── requirements
│   ├── ...
│   └── model_dependencies
│       └── {model_family}.txt
├── model_configurations
│   └── {model_family}
│       ├── config.json
│       └── model_weights.pt (optional)
└── serving_logic
    └── {model_family}.py
```

So in general, `requirements/model_dependencies` directory allows to place requirements file that will
be used to install additional dependencies for specific model serving (it will be helpful if
we wanted to have services with different models which may require contradictory dependencies).

At the same time - we need to provide model configuration under (`model_configurations/{model_family}/config.json`),
which has the following structure:
```json
{
  "model_family": "yolov5",
  "weights_file_name": "name_of_you_weights_file_located_in_the_same_dir",
  "factory_parameters": {
  }
}
```

`factory_parameters` key is purely dependent on model - for instance `yolov5` config:
```json
{
  "model_family": "yolov5",
  "weights_file_name": null,
  "factory_parameters": {
    "model_version": "yolov5s",
    "inference_size": 640
  }
}
```

It is important to provide specific inference module for our models. Example for [yolov5](./serving_logic/yolov5.py):
Each module of this kind must provide function to load model with the following signature:
```python
def load_XXX_model(
    factory_parameters: XXXFactoryParameters,
    device: torch.device,
    weights_path: Optional[str] = None,
) -> Tuple[ModelObject, InferenceFunction]:
    ...
```
which is meant to load model (according to `factory_parameters`) and assembly inference function
of this signature:
```python
ModelObject = Any  # this can be anything even, tensorflow if someone wanted
InferenceFunction = Callable[
    [ModelObject, List[np.ndarray], torch.device], List[InferenceResult]
]
```

Once such module is created - one should extend [serving_utils | load_model(...)](./serving_logic/serving_utils.py)
function:
```python
def load_model(
    context: Context, device: torch.device
) -> Tuple[ModelObject, InferenceFunction]:
    model_config = load_model_config(context=context)
    if model_config.model_family == YOLO_V5_FAMILY_NAME:
        from yolov5 import load_yolov5_model, YoloV5FactoryParameters # <- local import to be used here
        # [...]
        return load_yolov5_model(
            factory_parameters=yolo_parameters, device=device, weights_path=weights_path
        )
```
So - in essence - one needs to dispatch loading of specific modules to handle inference - when custom ones
are added.

Thanks to that structure - adding new models is just a matter of creating module that will load model
and handle inference, rest will be handled by pre-assembled repository components.

## How to get inference from your model via HTTP?
### Docker image build
GPU version:
```bash
repository_root$ docker build --build-arg MODEL_FAMILY=yolov5 -f ./docker/Dockerfile-gpu -t make-sense-serving-gpu .
```

CPU version:
```bash
repository_root$ docker build --build-arg MODEL_FAMILY=yolov5 -f ./docker/Dockerfile-cpu -t make-sense-serving-cpu .
```

### Run service
GPU version:
```bash
docker run -p 8080:8080 --runtime nvidia make-sense-serving-gpu
```
CPU version:
```bash
docker run -p 8080:8080 make-sense-serving-cpu
```

### Send request
```bash
curl -X POST http://127.0.0.1:8080/predictions/object_detector -F image=@path_to_your_image.jpg | jq
```

### Responses format
```json
{
  "image_height": 720,
  "image_width": 1280,
  "detections": [
    {
      "bounding_box": {
        "left_top_x": 0.5806957244873047,
        "left_top_y": 0.06714392768012153,
        "right_bottom_x": 0.8919973373413086,
        "right_bottom_y": 1
      },
      "prediction_details": {
        "confidence": 0.8798607587814331,
        "class_id": 0,
        "class_name": "person"
      }
    }
  ]
}
```

## Supported models
* [yolov5](https://github.com/ultralytics/yolov5)
* `retinanet_resnet50_fpn_v2` from `torchvision`

## Contribution guide

### :rotating_light: Repository setup
To initialize conda environment use
```bash
conda create -n MakeSenseServing python=3.9
conda activate MakeSenseServing
```

To install dependencies use (depending on `MODEL_FAMILY` you work on)
```bash
(MakeSenseServing) repository_root$ pip install -r requirements/requirements[-gpu].txt
(MakeSenseServing) repository_root$ pip install -r requirements/requirements-dev.txt
(MakeSenseServing) pip install -r build_dependencies/requirements-torchserve.txt
(MakeSenseServing) repository_root$ pip install -r build_dependencies/model_dependencies/${MODEL_FAMILY}.txt
```

To enable `pre-commit` use
```bash
(MakeSenseServing) repository_root$ pre-commit install
```

To run `pre-commit` check
```bash
(MakeSenseServing) repository_root$ pre-commit
```

To run tests, linter
```bash
(MakeSenseServing) repository_root$ pytest
(MakeSenseServing) repository_root$ black .
```
