#!/bin/sh

cd "$(dirname "$(dirname "$0")")" || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=tools/train.py  # Fixed the train file path

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
GPU=None

while getopts "p:d:c:n:w:g:r:" opt; do
  case $opt in
    p) PYTHON=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    c) CONFIG=$OPTARG ;;
    n) EXP_NAME=$OPTARG ;;
    w) WEIGHT=$OPTARG ;;
    r) RESUME=$OPTARG ;;
    g) GPU=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" ;;
  esac
done

if [ "$GPU" = 'None' ]; then
  GPU=$($PYTHON -c 'import torch; print(torch.cuda.device_count())')
fi

echo "Experiment Name: $EXP_NAME"
echo "Python Interpreter: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "Number of GPUs: $GPU"

EXP_DIR="$ROOT_DIR/exp/${DATASET}"
MODEL_DIR="${EXP_DIR}/model"
CONFIG_DIR="${ROOT_DIR}/configs/${DATASET}/${CONFIG}.py"

echo "==========> SETTING UP EXPERIMENT DIRECTORY <=========="
echo "Experiment Directory: $EXP_DIR"

if [ "$RESUME" = "true" ]; then
#  CONFIG_DIR="${EXP_DIR}/config.py"
  WEIGHT="${ROOT_DIR}/exp/${DATASET}/${CONFIG}/model/model_last.pth"
else
  mkdir -p "$MODEL_DIR"
fi

echo "Loading Config From: $CONFIG_DIR"

echo "==========> START TRAINING <=========="

TRAIN_CMD="$PYTHON $ROOT_DIR/$TRAIN_CODE --config-file $CONFIG_DIR --num-gpus $GPU --options save_path=$EXP_DIR"

if [ "$WEIGHT" != "None" ]; then
  TRAIN_CMD="$TRAIN_CMD resume=$RESUME weight=$WEIGHT"
fi

echo "Executing: $TRAIN_CMD"
eval $TRAIN_CMD
