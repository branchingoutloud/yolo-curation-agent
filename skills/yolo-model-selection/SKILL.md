---
name: yolo-model-selection
description: Select a YOLO model variant and starting hyperparameters given a
  deployment target and class count. Use when proposing a model size before
  training, or when training results suggest the current size is a poor fit.
license: MIT
---

# YOLO model selection

## Size vs. deployment target
| Target | Variant | Params | Typical use |
|---|---|---|---|
| Microcontroller / very constrained edge | YOLO11n | ~2.6M | Jetson Nano, RPi, mobile, <10ms budget |
| Edge GPU (Jetson Orin, mobile GPU) | YOLO11s | ~9.4M | Real-time on-device with headroom |
| Server GPU, moderate latency budget | YOLO11m | ~20M | Balanced accuracy/speed |
| Server GPU, accuracy-first | YOLO11l / YOLO11x | ~25M-57M | Offline or batch inference, max accuracy |

## Decision rule
1. Start from the deployment target row above.
2. If class count > 20 or classes are visually similar (fine-grained), move
   one size up from the target-implied default - small models under-fit
   fine-grained distinctions.
3. If the user needs real-time video (>=15 FPS) on the stated hardware, do not
   go above the target row's default regardless of class count; recommend
   more training data or augmentation instead of a bigger model.

## Default hyperparameters to propose
- imgsz: 640 (edge: consider 416 if latency-bound)
- epochs: 100 baseline, 150-300 for small/imbalanced datasets
- batch: largest that fits the sandbox GPU memory; halve if OOM
- patience (early stopping): 20-30

## When re-invoked after a poor training run
If eval results show broad underperformance (not just 1-2 classes), consider
recommending one size up before recommending more data - small models can be
data-starved in a way that looks like a data problem but is a capacity problem.
