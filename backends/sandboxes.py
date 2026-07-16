"""Sandbox backends for subagents that need real (GPU) compute via execute().

Two separate sandboxes, per §8 of the architecture doc:
- annotation: short-lived, CPU/small-GPU, just runs a zero-shot detector batch.
- training: longer-lived, GPU required, reused by eval-agent afterward so the
  trained weights are already local to that sandbox.

deepagents 0.6.12 does not ship a ModalSandbox — only SandboxBackendProtocol /
BaseSandbox to implement one against (plus LangSmithSandbox). Until the
training/annotation subagents are actually built (they're deprioritized behind
planning/sourcing/dataset), degrade to None so `langgraph dev` still boots —
same philosophy as tools/mcp_clients.py. A None backend means the subagent
runs on the default project backend with no execute() sandbox.
"""

import os

try:  # provided by a newer deepagents or a Modal integration package
    from deepagents.backends.sandbox import ModalSandbox
except ImportError:
    ModalSandbox = None
    print(
        "[sandboxes] ModalSandbox not available in this deepagents version - "
        "training/annotation subagents will run without an execute() sandbox"
    )

if ModalSandbox is not None:
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
else:
    training_sandbox_backend = None
    annotation_sandbox_backend = None
