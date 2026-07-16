"""Sandbox backends for subagents that need real (GPU) compute via execute().

Two separate sandboxes, per §8 of the architecture doc:
- annotation: short-lived, CPU/small-GPU, just runs a zero-shot detector batch.
- training: longer-lived, GPU required, reused by eval-agent afterward so the
  trained weights are already local to that sandbox.

Swap ModalSandbox for DaytonaSandbox (or another deepagents-supported sandbox)
here if you're not using Modal - nothing outside this file needs to change.
"""

import os

from deepagents.backends.sandbox import ModalSandbox

training_sandbox_backend = ModalSandbox(
    image=os.environ.get("TRAINING_SANDBOX_IMAGE", "ultralytics/ultralytics:latest-python"),
    gpu=os.environ.get("TRAINING_SANDBOX_GPU", "A10G"),
    timeout=int(os.environ.get("TRAINING_SANDBOX_TIMEOUT", str(60 * 45))),
)

annotation_sandbox_backend = ModalSandbox(
    image=os.environ.get("ANNOTATION_SANDBOX_IMAGE", "python:3.11-slim"),
    gpu=os.environ.get("ANNOTATION_SANDBOX_GPU", "T4"),
    timeout=int(os.environ.get("ANNOTATION_SANDBOX_TIMEOUT", str(60 * 15))),
)
