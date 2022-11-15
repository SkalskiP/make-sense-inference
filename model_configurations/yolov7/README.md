# YOLO V7 deployment
As original [YOLO V7 repository](https://github.com/WongKinYiu/yolov7) is not pip-installable and
TorchHub is not really supported, in order to deploy this model - it needs to be exported at first into
TorchScript format.

Pre-trained model weights were already exported and published in the target format - please
[see release package](https://github.com/SkalskiP/make-sense-inference/releases/tag/yolo_v7_exported_models).
**Important note - all models are exported to support (640, 640) inference size - and serving handler is
adjusted to standardise all images to that dimensions size.**

## Config structure
```json
{
  "model_family": "yolov7",
  "weights_file_name": "yolov7-tiny.torchscript.pt",
  "factory_parameters": {
    "inference_size": [640, 640],
    "classes": ["class_a", "class_b"],
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45
  }
}
```
In particular:
* `weights_file_name` must be given - one must place proper weights file in this directory - example ones to be
downloaded from [release package](https://github.com/SkalskiP/make-sense-inference/releases/tag/yolo_v7_exported_models)
* `inference_size` dictate resizing for inference (results will be converted back into original proportions)
* `classes` list determine class mapping
* `confidence_threshold` and `iou_threshold` are used as NMS parameters


## What if I want to deploy my own model
At first - train your model using [YOLO V7 repository](https://github.com/WongKinYiu/yolov7). Having that done
it is already great news, as the conversion to TorchScript should be by far the easiest conversion to be done
(among others, like ONNX, CoreML, TensorRT etc.) - as this is Torch-native way of porting models.

In fact - even in [YOLO V7 repository](https://github.com/WongKinYiu/yolov7) there is conversion method. Take a look
at [export.py](https://github.com/WongKinYiu/yolov7/blob/main/export.py)

In fact - this is the only important piece of code.
```python
    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run
    if opt.include_nms:
        model.model[-1].include_nms = True
        y = None

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)
```
Nothing else is needed.

Unfortunately - there is an issue with dependency resolution - in fact probably author of the repo was using them
interchangeably - per use case. To convert model we need:
```
# Export --------------------------------------
coremltools>=4.1  # CoreML export
onnx>=1.9.0  # ONNX export
onnx-simplifier>=0.3.6  # ONNX simplifier
#scikit-learn==0.19.2  # CoreML quantization
#tensorflow>=2.4.1  # TFLite export
#tensorflowjs>=3.9.0  # TF.js export
#openvino-dev  # OpenVINO export
```
This is minimal set of uncommented dependencies (three first) that makes all imports to work and at the same
time are possible to be resolved. **Please convert model in different env that is used for the sake of this repository -
exported model should be portable.**

At the end of the day we have the following command to be used:
```bash
python export.py --weights <your_weights_path> --grid --end2end --img-size 640 640
```
