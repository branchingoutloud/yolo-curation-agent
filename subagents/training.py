import os

from deepagents.middleware.subagents import CompiledSubAgent

from subagents.sandbox_subagent import build_sandbox_subagent
from tools.model_builder import build_model

# TRAINING_AGENT_MODEL is an OPTIONAL override - unset by default, so
# training-agent inherits ORCHESTRATOR_MODEL. Its job is mostly mechanical -
# pick a model size per the skill's rule, launch execute(), append status
# lines - so a lighter model is a reasonable override to set explicitly
# rather than spending the same budget as sourcing/dataset-agent's
# tool-calling load.

# Built via build_sandbox_subagent (subagents/sandbox_subagent.py), NOT a
# plain SubAgent dict with a "backend" field - that field is silently
# ignored by deepagents==0.6.12's subagent-building loop (see
# tools/dataset_builder.py's module docstring / CLAUDE.md's Known Stubs).
# This is instead its own separately-compiled create_deep_agent(backend=
# sandbox_backend, ...) graph wrapped as a CompiledSubAgent, which DOES get
# a real execute() tool.

# TRAINING_BACKEND=local (default, backends/sandboxes.py) roots
# LocalShellBackend at the SAME RUN_ARTIFACTS_DIR real disk project_backend
# uses - dataset/ and model_choice.json are already there, no upload needed,
# and whatever this subagent writes back is immediately visible on real disk
# too, no download needed. TRAINING_BACKEND=modal is a genuinely separate
# remote filesystem, so THAT mode still needs the copy-in/copy-out bridge.
_USING_MODAL = os.environ.get("TRAINING_BACKEND", "local") == "modal"


def _resolve_model():
    # Unlike a declarative SubAgent dict (which deepagents' own subagent-
    # building loop falls back to ORCHESTRATOR_MODEL for automatically via
    # `spec.get("model", model)`), this subagent is its own independent
    # create_deep_agent(...) call - it has no automatic visibility into the
    # orchestrator's resolved model, so an unset TRAINING_AGENT_MODEL must
    # be resolved against ORCHESTRATOR_MODEL explicitly here, or it silently
    # falls back to deepagents' own built-in default (Anthropic) and fails
    # without ANTHROPIC_API_KEY set - the same gotcha documented in
    # CLAUDE.md for standalone test scripts.
    model_spec = os.environ.get("TRAINING_AGENT_MODEL") or os.environ.get(
        "ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5"
    )
    return build_model(model_spec)


# Path convention below is intentionally DIFFERENT for tool calls vs. shell
# commands, and this bit a real live run before being fixed:
#   - ls/read_file/write_file/edit_file (this subagent's own standalone
#     LocalShellBackend, NOT routed through project_backend's CompositeBackend
#     - there is no "/workspace/" prefix to strip here at all) resolve a
#     leading "/" as the virtual root = root_dir directly, confirmed against
#     deepagents' own _resolve_path: `(self.cwd / vpath.lstrip("/")).resolve()`.
#     So "/dataset/data.yaml" is correct for those tools.
#   - execute() shell commands are NOT virtual-path-resolved at all - the
#     command string is passed straight to the OS shell with cwd=root_dir
#     already set. On Windows, a leading "/" gets read as "root of the
#     current drive", not "relative to cwd" - confirmed live: a real
#     training-agent run got FileNotFoundError on "/workspace/dataset/
#     data.yaml" for exactly this reason, correctly self-diagnosed by the
#     model ("environment uses Windows path style but expects virtual
#     paths") but never actually fixed by it. Plain relative paths (no
#     leading slash) avoid this entirely, verified directly against the real
#     backend object before rewriting this prompt.
# Modal mode (a real Linux container) doesn't have this problem - a leading
# "/" is a real, correct absolute path there - but this hasn't been
# re-verified against a live Modal run, so treat the _USING_MODAL branch
# below as reasoned-through, not live-tested.
_LOCAL_PROMPT = (
    "This is a smoke test that the pipeline works end to end on a local CPU - not "
    "a real training run. Follow these exact steps in order, using as FEW execute() "
    "calls as possible (2-3 total, not a back-and-forth loop of checks).\n\n"
    "IMPORTANT path convention - these are NOT the same: when calling ls/read_file/"
    "write_file/edit_file, paths start with / (e.g. /dataset/data.yaml). Inside an "
    "execute() shell command string, paths must NOT start with / - use a plain "
    "relative path instead (e.g. dataset/data.yaml, not /dataset/data.yaml) - a "
    "leading slash in a shell command resolves to the current drive's root on this "
    "Windows machine, not the working directory, and will fail with "
    "FileNotFoundError.\n\n"
    "IMPORTANT - always redirect a training command's own output to a log file "
    "(e.g. `... > train_log.txt 2>&1`) rather than relying on execute()'s own "
    "returned output. This environment's execute() can fail to capture ultralytics' "
    "console output at all (a Windows console-encoding limitation, not something you "
    "can fix), which looks like \"no output\" or a vague failure even when training "
    "actually succeeded - do NOT interpret empty/missing execute() output as a real "
    "failure (and specifically not as \"no GPU/CUDA available\" - this box may or may "
    "not have a GPU and it doesn't matter, device=cpu is used regardless). Always "
    "check the redirected log file (via read_file, not execute()) to see what "
    "actually happened before concluding anything failed.\n\n"
    "Step 0 - use read_file (NOT execute()) on /dataset/data.yaml BEFORE attempting "
    "anything else. If this errors (file doesn't exist), STOP immediately and report "
    "exactly that: dataset-agent has not actually produced a dataset yet at this path. "
    "Do NOT guess about missing dependencies, missing GPU, or any other cause when the "
    "real problem is simply that the input file isn't there yet - quote the actual "
    "read_file error in your report. Do NOT use write_file to create or patch "
    "/dataset/data.yaml yourself as a workaround - you have no way to know what images "
    "actually exist or what split ratios are real, so a hand-written data.yaml would "
    "point at directories with no real images in them and only produce a more confusing "
    "failure later. Stop and report the gap; do not paper over it.\n\n"
    "Step 1 - ONE execute() call, combined with shell operators, to check "
    "ultralytics is present and install it only if missing - do not run these as "
    "two separate calls, do not re-check afterward:\n"
    "  python -c \"import ultralytics\" || pip install ultralytics\n\n"
    "Step 2 - ONE execute() call to run training, with output redirected to a log "
    "file. This call BLOCKS until training finishes (or times out) - you do NOT "
    "need to poll or check on it while it's running, so do not issue any other "
    "execute() calls in between. Use EXACTLY this command, including `project=%cd%"
    "\\runs` (do not use `project=runs` - ultralytics has its own cached global "
    "settings for where a bare relative project path resolves to, which is NOT "
    "this working directory and has produced wrong/nested output paths before; "
    "`%cd%` is cmd.exe's own current-directory variable and always resolves "
    "correctly). Do not increase epochs/imgsz or use a larger model - the goal is "
    "fast and small, not accurate:\n"
    "  yolo detect train data=dataset/data.yaml model=yolo11n.pt epochs=2 "
    "imgsz=320 device=cpu project=%cd%\\runs name=train exist_ok=True > "
    "train_log.txt 2>&1\n\n"
    "Step 3 - use read_file (NOT execute()) on /train_log.txt to see what actually "
    "happened, and read whatever ultralytics actually produced (its own results.csv "
    "under runs/train/ - inspect it via read_file too, e.g. /runs/train/results.csv). "
    "Then write exactly two files yourself based on that real output - do not "
    "fabricate numbers if the run failed, write the actual error from the log "
    "instead: /runs/train/status.md (a short human-readable summary - what ran, "
    "whether it succeeded) and /runs/train/metrics.json (a simple {metric_name: "
    "value} shape from ultralytics' real results). Trained weights stay wherever "
    "ultralytics put them under runs/train/weights/ for eval-agent to read directly "
    "- don't move them.\n\n"
    "Use EXACTLY the `yolo detect train ...` CLI invocation shown above - do NOT "
    "substitute `python -m ultralytics.yolo ...`, `python -c \"from ultralytics import "
    "YOLO; ...\"`, or any other invocation you construct yourself. The `yolo` console "
    "script is already on PATH in this environment (confirmed - do not second-guess "
    "this); a different invocation style has previously failed with `No module named "
    "ultralytics.yolo` (an outdated/wrong module path, not a real missing-dependency "
    "problem) purely because it didn't match this exact command.\n\n"
    "If step 1 or step 2's log shows a genuine failure, do not retry more than "
    "once - your final message MUST quote the actual line(s) from the log file "
    "verbatim, not a paraphrased guess. Never claim ultralytics/GPU/CUDA is "
    "unavailable unless the log you actually read literally says so - if you "
    "haven't read train_log.txt yet, read it before writing your final message."
)

_MODAL_PROMPT = (
    "Read /workspace/dataset/data.yaml and, if present, the confirmed model size from "
    "/workspace/model_choice.json. Launch training via execute() - e.g. `yolo detect "
    "train data=/workspace/dataset/data.yaml model=yolo11n.pt epochs=2 imgsz=320 "
    "project=/workspace/runs name=train exist_ok=True`. Do not increase epochs/imgsz or "
    "use a larger model. Every few epochs, append a short status line to "
    "/workspace/runs/train/status.md. On completion, write /workspace/runs/train/"
    "metrics.json from ultralytics' real results, and leave trained weights under "
    "/workspace/runs/train/weights/ for eval-agent to reuse directly."
)


def build_training_agent(sandbox_backend) -> CompiledSubAgent:
    return build_sandbox_subagent(
        name="training-agent",
        description="Selects a YOLO model size and runs a small local training smoke run.",
        system_prompt=_MODAL_PROMPT if _USING_MODAL else _LOCAL_PROMPT,
        sandbox_backend=sandbox_backend,
        model=_resolve_model(),
        upload_paths=(
            ["/workspace/dataset", "/workspace/model_choice.json"] if _USING_MODAL else []
        ),
        download_paths=(
            ["/workspace/runs/train/status.md", "/workspace/runs/train/metrics.json"]
            if _USING_MODAL
            else []
        ),
    )
