# MEGDet

MEGDet is a YOLO-based multimodal object detection project built on the Ultralytics framework. The project focuses on RGB-IR fusion detection for remote-sensing and UAV scenes, with training scripts and dataset configuration examples for DroneVehicle and VEDAI.

## Project Structure

```text
MEGDet/
├── ultralytics/                 # Modified Ultralytics/YOLO source code
├── DroneVehicle_merge.yaml      # DroneVehicle RGB-IR dataset configuration
├── train_drone.py               # Training script for DroneVehicle
├── train_vedai.py               # Training script for VEDAI
└── README.md
```

## Main Features

- YOLO-based object detection and oriented bounding box detection.
- RGB and infrared image fusion input support.
- Dataset configuration for DroneVehicle RGB-IR training.
- Training scripts for DroneVehicle and VEDAI experiments.
- Custom model configuration files under `ultralytics/cfg/models/11/`.

## Training

Install the required Python environment first, then run the training scripts from the project root.

### DroneVehicle

```bash
python train_drone.py
```

The DroneVehicle script uses:

- dataset config: `DroneVehicle_merge.yaml`
- task: `obb`
- image size: `640`
- epochs: `200`
- output directory: `runs/test/yolov11n_fusion`

### VEDAI

```bash
python train_vedai.py
```

The VEDAI script uses:

- dataset config: `VEDAI_merge.yaml`
- task: `detect`
- image size: `1024`
- epochs: `200`
- output directory: `runs/VEDAI/yolov11s_fusion`

## Dataset Notes

The dataset paths in the YAML files and scripts are written for the author's local/server environment. Before training on a new machine, update the dataset paths and model configuration paths to match your own environment.

For example, `DroneVehicle_merge.yaml` expects RGB and IR image folders such as:

```text
/data/DroneVehicle/merge/train/ir/images
/data/DroneVehicle/merge/train/rgb/images
```

## Repository

This repository contains the experimental code for MEGDet and related RGB-IR fusion detection experiments.
