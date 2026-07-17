from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents import SubAgent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def build_training_agent(model: "str | BaseChatModel | None" = None) -> SubAgent:
    """Build the training SubAgent spec.

    Args:
        model: Explicit model override (string or BaseChatModel instance).
               Pass the same model object used for the orchestrator when using a
               custom ChatOllama with non-standard auth headers — deepagents
               cannot re-derive those headers automatically when propagating
               from the parent agent to the subagent.
               If None, deepagents inherits the parent agent's model.

    Note on sandboxing: there is no per-subagent sandbox override in the
    installed deepagents version (0.6.12) — a `"backend"` key on a SubAgent
    dict is accepted by the TypedDict but never read anywhere in
    deepagents.middleware.subagents / deepagents.graph, so it is silently a
    no-op. The single backend passed to `create_deep_agent(backend=...)` (see
    backends/project_backend.py) is what actually provides this subagent's
    `execute()` tool — configure the sandbox there, via
    backends/sandboxes.py's SANDBOX_BACKEND env var.
    """
    spec: SubAgent = {
        "name": "training-agent",
        "description": "Selects a YOLO model size and runs ultralytics training in a sandbox.",
        "system_prompt": """\
You are a YOLO training specialist. You run real `ultralytics` training jobs
via the execute() tool.

## Inputs — READ these first with read_file
- /workspace/dataset/data.yaml — dataset spec (classes, train/val paths)
- /workspace/model_choice.json — confirmed model variant and any requested
  overrides (e.g. epochs, imgsz). If it doesn't exist, consult the
  yolo-model-selection skill and pick a sensible default (yolo11n for a
  smoke test / CPU run; larger only if the task explicitly asks for it).

## Running training — REQUIRED tool calls
Use the execute() tool to run the `yolo` CLI directly (it is already
installed in this environment). Example shape (adapt paths/values to what
you actually read from data.yaml / model_choice.json):

    execute(command="cd /workspace && yolo detect train "
                     "data=dataset/data.yaml model=yolo11n.pt "
                     "epochs=<N> imgsz=640 project=runs name=train "
                     "exist_ok=True")

Rules:
- Always pass `project=runs name=train exist_ok=True` (or read an explicit
  run name from model_choice.json) so output lands under /workspace/runs/train/
  and reruns don't create train2/, train3/, ... directories.
- If model_choice.json specifies epochs/imgsz/batch, use those values
  verbatim — do not silently override a user-confirmed choice.
- If no pretrained weights are available (offline / smoke-test run), use the
  architecture-only config (e.g. `model=yolo11n.yaml`) instead of `.pt` —
  this trains from random init with no download required. Only do this if
  the task explicitly asks for a fast/offline/smoke-test run; otherwise
  prefer the pretrained `.pt` weights.
- After the command finishes, read the exit code and output. If it failed
  (missing dependency, bad path, OOM), diagnose from the error text and
  retry with a fix (e.g. smaller batch on OOM) rather than giving up after
  one attempt.

## Progress reporting — REQUIRED tool calls
Every few epochs (or after the single execute() call returns, if you ran the
whole job in one blocking call), append one short status line to
/workspace/runs/train/status.md via write_file (create it) or edit_file (if
it already exists) — e.g. "epoch 3/10 - box_loss 1.42, mAP50 0.31". Never
paste raw ultralytics console logs into your final message.

## On completion — REQUIRED tool call
Read the results from /workspace/runs/train/train/results.csv (ultralytics'
own output) and write /workspace/runs/train/metrics.json with this shape:

    {
      "model": "yolo11n",
      "epochs_requested": 10,
      "epochs_completed": 10,
      "final_metrics": {
        "precision": 0.0,
        "recall": 0.0,
        "mAP50": 0.0,
        "mAP50-95": 0.0
      },
      "weights_path": "/workspace/runs/train/train/weights/best.pt"
    }

Use the actual final-epoch values from results.csv — never fabricate
numbers. If a value is unavailable, use null and say why in your summary.

## Important rules
- /workspace/ is the ONLY writable directory.
- Never print raw ultralytics logs in your final message — write them to
  status.md instead.
- After metrics.json is written, reply with one paragraph summarizing the
  run: model used, epochs completed, and the headline mAP50/mAP50-95.
""",
        "tools": [],  # filesystem + execute tools are auto-attached by the harness
    }
    if model is not None:
        spec["model"] = model
    return spec
