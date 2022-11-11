#!/bin/bash
set -e

EXTRA_FILES=$(python list_serving_modules.py)

torch-model-archiver \
  --model-name object_detector \
  --version base \
  --export-path $MODEL_DIR \
  --handler $SERVING_LOGIC_DIR/handler.py \
  --extra-files $EXTRA_FILES
