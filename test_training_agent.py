"""
Test: Orchestrator -> task("training-agent") -> runs/train/metrics.json + status.md

Flow being tested:

    User
       |
       v
    Orchestrator  (create_deep_agent, stripped to training only)
       |
    task("training-agent")
       |
       v
    Training Agent  --execute()--> real `yolo` CLI training run (sandbox)
       |
       +-- runs/train/status.md
       +-- runs/train/metrics.json

The orchestrator is given ONLY the training-agent as a subagent and is
instructed to delegate immediately (no HITL gates, no other subagents).
This exercises task() delegation AND the execute() sandbox tool end to end -
dataset-agent's job (assembling dataset/data.yaml) is out of scope for this
test, so a tiny synthetic YOLO-format dataset is staged directly as a fixture
before the orchestrator runs, exactly like a real dataset-agent output would
look.

Uses `model=yolo11n.yaml` (architecture only, random init) instead of a
pretrained `.pt` checkpoint so the run never needs a network download -
this is a smoke test of the pipeline mechanics, not of model accuracy.

Exit codes:
    0  all assertions pass
    1  one or more assertions failed
    2  environment misconfigured
"""

import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# Disable LangSmith so traces do not require a key during testing
# ------------------------------------------------------------------

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ------------------------------------------------------------------
# Validate environment
# ------------------------------------------------------------------

required = [
    "OLLAMA_API_KEY",
    "OLLAMA_BASE_URL",
]

missing = [k for k in required if not os.getenv(k)]
if missing:
    for k in missing:
        print(f"[ERROR] Missing environment variable: {k}")
    sys.exit(2)

# ------------------------------------------------------------------
# LLM  --  reuse the same Ollama Cloud setup used elsewhere
# ------------------------------------------------------------------

from langchain_ollama import ChatOllama

LLM_MODEL = os.environ.get("TRAINING_TEST_MODEL", "gpt-oss:20b")

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=os.environ["OLLAMA_BASE_URL"],
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"
        }
    },
)

print(f"[ok] LLM      : {LLM_MODEL} via Ollama Cloud")

# ------------------------------------------------------------------
# Backend  --  same shared virtual filesystem + sandbox as production
#   /workspace/ --> run_artifacts/ on real disk, execute() runs for real
#   (SANDBOX_BACKEND=local by default - see backends/sandboxes.py)
# ------------------------------------------------------------------

from backends.project_backend import project_backend, RUN_ARTIFACTS_DIR

RUN_DIR = Path(RUN_ARTIFACTS_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"[ok] Backend  : /workspace/ -> {RUN_DIR} (execute() sandbox: {os.environ.get('SANDBOX_BACKEND', 'local')})")

# ------------------------------------------------------------------
# Clean previous outputs so assertions are always fresh
# ------------------------------------------------------------------

DATASET_DIR = RUN_DIR / "dataset"
RUNS_DIR = RUN_DIR / "runs"
MODEL_CHOICE_FILE = RUN_DIR / "model_choice.json"

for stale_dir in (DATASET_DIR, RUNS_DIR):
    if stale_dir.exists():
        shutil.rmtree(stale_dir)
        print(f"     Removed stale {stale_dir.relative_to(RUN_DIR)}/")

if MODEL_CHOICE_FILE.exists():
    MODEL_CHOICE_FILE.unlink()
    print("     Removed stale model_choice.json")

# ------------------------------------------------------------------
# Fixture: a tiny synthetic YOLO-format dataset (stands in for
# dataset-agent's output, which is out of scope for this test) plus a
# confirmed model_choice.json (stands in for the orchestrator's Gate 2).
# ------------------------------------------------------------------

from PIL import Image, ImageDraw

IMG_SIZE = 64  # small + a multiple of 32 (yolo's max stride) -> fast CPU epoch
CLASS_NAME = "widget"


def _make_image_with_box(path: Path, seed: int) -> tuple[int, int, int, int]:
    """Draw a solid rectangle on a noise-free background; return its pixel bbox."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    x0, y0 = 10 + (seed % 3) * 5, 10 + (seed % 4) * 3
    x1, y1 = x0 + 24, y0 + 24
    draw.rectangle([x0, y0, x1, y1], fill=(220, 60, 60))
    img.save(path)
    return x0, y0, x1, y1


def _write_yolo_label(path: Path, bbox: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bbox
    cx = ((x0 + x1) / 2) / IMG_SIZE
    cy = ((y0 + y1) / 2) / IMG_SIZE
    w = (x1 - x0) / IMG_SIZE
    h = (y1 - y0) / IMG_SIZE
    path.write_text(f"0 {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n", encoding="utf-8")


def stage_synthetic_dataset() -> None:
    for split, count in (("train", 6), ("val", 2)):
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            bbox = _make_image_with_box(img_dir / f"{split}_{i}.jpg", seed=i)
            _write_yolo_label(lbl_dir / f"{split}_{i}.txt", bbox)

    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n  0: {CLASS_NAME}\n",
        encoding="utf-8",
    )


stage_synthetic_dataset()
print(f"[ok] Fixture  : synthetic dataset staged at {DATASET_DIR.relative_to(RUN_DIR)}/ (6 train / 2 val, class '{CLASS_NAME}')")

MODEL_CHOICE_FILE.write_text(
    json.dumps(
        {
            "model": "yolo11n.yaml",  # architecture only - no pretrained-weight download
            "epochs": 1,
            "imgsz": IMG_SIZE,
            "batch": 4,
            "note": "Smoke test run - 1 epoch, tiny synthetic dataset, no internet required.",
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("[ok] Fixture  : model_choice.json written (yolo11n.yaml, 1 epoch)")

# ------------------------------------------------------------------
# Training subagent  --  import from the real module
# ------------------------------------------------------------------

from subagents.training import build_training_agent

# Pass the same llm explicitly so deepagents uses it directly for the
# subagent instead of trying to re-derive it from the parent orchestrator.
training_agent = build_training_agent(model=llm)

print(f"[ok] Subagent : {training_agent['name']}")

# ------------------------------------------------------------------
# Minimal orchestrator  --  only knows about training-agent
#
# No HITL gates, no planning/sourcing/annotation/dataset/eval subagents.
# Goal: purely test the User -> Orchestrator -> task() -> Agent -> execute()
# path.
# ------------------------------------------------------------------

from deepagents import create_deep_agent

# Virtual backend path, not a host filesystem path — see
# backends/project_backend.py's "/skills/" route.
SKILLS_DIR = "/skills"

TEST_ORCHESTRATOR_PROMPT = """\
You are a test orchestrator. Your ONLY task is:

1. Call task("training-agent") with the user's request.
2. Wait for the training-agent to finish and return its summary.
3. Reply with a short confirmation that includes the summary.

Do NOT call request_approval. Do NOT do anything else.
Do not run training yourself -- the training-agent handles that.
"""

orchestrator = create_deep_agent(
    model=llm,
    system_prompt=TEST_ORCHESTRATOR_PROMPT,
    subagents=[training_agent],  # only training-agent is available
    tools=[],  # no approval tool needed for this test
    skills=[SKILLS_DIR],
    backend=project_backend,
)

print("[ok] Orchestrator built (training-agent only)")
print()

# ------------------------------------------------------------------
# User prompt
# ------------------------------------------------------------------

USER_PROMPT = """\
The dataset is ready at /workspace/dataset/data.yaml and the model choice is
confirmed in /workspace/model_choice.json. Run training now exactly as
configured there (it's a fast 1-epoch smoke test, not a real training run --
do not increase epochs or swap in a bigger model). When it's done, write the
status and metrics files as instructed.
"""

print("=" * 60)
print("USER PROMPT")
print("=" * 60)
print(USER_PROMPT.strip())
print("=" * 60)
print()
print("Running Orchestrator --> Training Agent ... (this runs a real training job, may take a minute or two)")
print()

# ------------------------------------------------------------------
# Invoke the orchestrator  (User -> Orchestrator -> task -> Agent -> execute())
# ------------------------------------------------------------------

response = orchestrator.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": USER_PROMPT,
            }
        ]
    },
    config={"recursion_limit": 100},
)

final_message = response["messages"][-1].content

print()
print("=" * 60)
print("ORCHESTRATOR FINAL RESPONSE")
print("=" * 60)
print(final_message)
print("=" * 60)

# ------------------------------------------------------------------
# Assertions
# ------------------------------------------------------------------

errors: list[str] = []
warnings: list[str] = []

status_file = RUNS_DIR / "train" / "status.md"
metrics_file = RUNS_DIR / "train" / "metrics.json"

# 1. status.md must exist and be non-empty
if not status_file.exists():
    errors.append(f"FAIL: {status_file.relative_to(RUN_DIR)} was not created under /workspace/")
elif len(status_file.read_text(encoding="utf-8").strip()) == 0:
    errors.append(f"FAIL: {status_file.relative_to(RUN_DIR)} exists but is empty")

# 2. metrics.json must exist and be valid JSON
metrics_data = None
if not metrics_file.exists():
    errors.append(f"FAIL: {metrics_file.relative_to(RUN_DIR)} was not created under /workspace/")
else:
    try:
        metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"FAIL: metrics.json is not valid JSON -- {exc}")

# 3. metrics.json should look like the shape requested in the system prompt
# (soft check -- exact keys are the model's judgment call, not enforced)
if isinstance(metrics_data, dict):
    content_str = json.dumps(metrics_data).lower()
    if "map" not in content_str and "precision" not in content_str and "recall" not in content_str:
        warnings.append("WARN: metrics.json has no recognizable metric field (mAP/precision/recall) -- check its content manually")

# 4. Real training must actually have run: ultralytics writes its own
# results.csv / weights under runs/**/ -- this is the strongest signal that
# execute() actually launched `yolo`, not just that the LLM wrote JSON.
ultralytics_outputs = list(RUNS_DIR.rglob("results.csv")) + list(RUNS_DIR.rglob("*.pt"))
if not ultralytics_outputs:
    errors.append(
        "FAIL: no results.csv or .pt weights found anywhere under runs/ -- "
        "execute() likely never launched a real `yolo` training run"
    )
else:
    found = ", ".join(str(p.relative_to(RUN_DIR)) for p in ultralytics_outputs[:5])
    print(f"[ok] Found real ultralytics output: {found}")

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------

print()

if warnings:
    for w in warnings:
        print(f"  * {w}")
    print()

if errors:
    print("=" * 60)
    print("TEST FAILED")
    print("=" * 60)
    for e in errors:
        print(f"  * {e}")
    print()
    sys.exit(1)

print("=" * 60)
print("TEST PASSED")
print("=" * 60)
print()

print("-- status.md " + "-" * 47)
print(status_file.read_text(encoding="utf-8"))
print()
print("-- metrics.json " + "-" * 43)
print(json.dumps(metrics_data, indent=2))
