import os

from deepagents.middleware.subagents import CompiledSubAgent

from subagents.sandbox_subagent import build_sandbox_subagent
from tools.model_builder import build_model

# EVAL_AGENT_MODEL is an OPTIONAL override - unset by default, so eval-agent
# inherits ORCHESTRATOR_MODEL. Diagnosing *why* a class underperforms (not
# just which metric is low) is the one genuinely reasoning-heavy step here,
# but the mechanical parts (confusion matrix via execute(), writing the two
# output files) don't need the orchestrator's full-size model - a lighter
# override is a reasonable default to set explicitly, same as training-agent.

# Built via build_sandbox_subagent, same reasoning as training-agent (see
# subagents/training.py) - a separately-compiled create_deep_agent(backend=
# sandbox_backend) graph wrapped as a CompiledSubAgent, not a plain SubAgent
# dict (whose "backend" field deepagents==0.6.12 silently ignores). No
# upload_paths ever needed here regardless of backend: `sandbox_backend` is
# the SAME sandbox instance passed to build_training_agent (see agent.py/
# backends/sandboxes.py's process-lifetime singleton), so the trained
# weights training-agent already produced are still present there - re-
# uploading would be redundant. download_paths for eval_report.md/
# weak_classes.json only matters in TRAINING_BACKEND=modal mode (a genuinely
# separate remote filesystem); under the local default they're already on
# real disk, so downloading would just copy a file onto itself - harmless
# but unnecessary, skipped the same way training.py skips its bridge locally.


_USING_MODAL = os.environ.get("TRAINING_BACKEND", "local") == "modal"


def _resolve_model():
    # See subagents/training.py's _resolve_model - same reasoning: this is
    # its own independent create_deep_agent(...) call, not a declarative
    # SubAgent dict, so it doesn't automatically inherit ORCHESTRATOR_MODEL
    # the way deepagents' own subagent-building loop would.
    model_spec = os.environ.get("EVAL_AGENT_MODEL") or os.environ.get(
        "ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5"
    )
    return build_model(model_spec)


# See subagents/training.py's identical note on why the path convention below
# differs by tool: ls/read_file/write_file/edit_file (this subagent's own
# standalone LocalShellBackend) take a leading "/" = virtual root = root_dir;
# execute() shell command strings must NOT have a leading "/" - on Windows
# that resolves to the current drive's root, not the working directory, and
# fails with FileNotFoundError (confirmed live on training-agent, same bug,
# same fix).
_LOCAL_PROMPT = (
    "Read /runs/train/metrics.json (write_file/read_file convention: leading slash) "
    "and inspect the trained weights under runs/train/weights/ (already here - "
    "training-agent just ran in this same environment). When referencing these paths "
    "inside an execute() shell command instead (e.g. running supervision/python code), "
    "use the relative form with NO leading slash: runs/train/weights/best.pt, not "
    "/runs/train/weights/best.pt - a leading slash in a shell command resolves to the "
    "current drive's root on this Windows machine, not the working directory.\n\n"
    "IMPORTANT - always redirect any execute() command that runs Python/supervision code "
    "to a log file (e.g. `... > eval_log.txt 2>&1`) rather than relying on execute()'s own "
    "returned output. This environment's execute() can fail to capture console output at "
    "all (a Windows console-encoding limitation, not something you can fix), which looks "
    "like \"no output\" even when the code ran fine - do NOT interpret empty/missing "
    "execute() output as a real failure. Check the redirected log file via read_file (not "
    "execute()) to see what actually happened.\n\n"
    "Use the `supervision` library via execute() to build a confusion matrix and sample "
    "failure crops. Consult the cv-eval-and-iteration skill for how to read mAP50/"
    "mAP50-95 and confusion-matrix patterns. For each underperforming class, inspect its "
    "sample count, source annotation quality, and failure crops, and record a specific "
    "likely cause - not just the metric. Using write_file (leading slash), write "
    "/eval_report.md (human-readable) and /weak_classes.json (structured: class name, "
    "metric, sample count, likely cause) so the orchestrator can decide whether to "
    "re-source, re-annotate, or just retune."
)

_MODAL_PROMPT = (
    "Read /workspace/runs/train/metrics.json and the trained weights under "
    "/workspace/runs/train/weights/ (already in this sandbox - training-agent just ran "
    "here). Use the `supervision` library via execute() to build a confusion matrix and "
    "sample failure crops. Consult the cv-eval-and-iteration skill for how to read "
    "mAP50/mAP50-95 and confusion-matrix patterns. For each underperforming class, "
    "inspect its sample count, source annotation quality, and failure crops, and record "
    "a specific likely cause - not just the metric. Write /workspace/eval_report.md "
    "(human-readable) and /workspace/weak_classes.json (structured: class name, metric, "
    "sample count, likely cause) so the orchestrator can decide whether to re-source, "
    "re-annotate, or just retune."
)


def build_eval_agent(sandbox_backend) -> CompiledSubAgent:
    return build_sandbox_subagent(
        name="eval-agent",
        description=(
            "Analyzes a completed training run: confusion matrix, per-class "
            "metrics, and a best-guess diagnosis for each weak class (too few "
            "images, high visual variability, likely label noise, or a class the "
            "zero-shot annotator struggled with). Does NOT decide the next "
            "action - that's the orchestrator's call based on this diagnosis."
        ),
        system_prompt=_MODAL_PROMPT if _USING_MODAL else _LOCAL_PROMPT,
        sandbox_backend=sandbox_backend,
        model=_resolve_model(),
        upload_paths=[],
        download_paths=(
            ["/workspace/eval_report.md", "/workspace/weak_classes.json"] if _USING_MODAL else []
        ),
    )
