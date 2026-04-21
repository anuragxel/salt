#!/bin/bash -e

DATASET_DIR=$1
USER_ID=$(id -u)

if [ -z "$DATASET_DIR" ]; then
  echo "Usage: $0 <dataset_dir>"
  exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
  echo "Error: Directory $DATASET_DIR does not exist."
  exit 1
fi

echo "Running pre-processing on dataset in $DATASET_DIR"

docker run --rm -it \
    -v "$DATASET_DIR":/dataset \
    --runtime nvidia \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    --env=USER_ID=$USER_ID \
    --gpus 1 \
    --name salt andreaostuni/salt:salt-cuda-11.8-base \
    /bin/bash -c "python3 /root/segment-anything/extract_embeddings.py --dataset-path /dataset --checkpoint-path /root/segment-anything/sam_vit_h_4b8939.pth && \
                  python3 /root/segment-anything/generate_onnx.py --dataset-path /dataset --checkpoint-path /root/segment-anything/sam_vit_h_4b8939.pth --onnx-models-path /dataset/models && \
                  chown -R $USER_ID:$USER_ID /dataset"