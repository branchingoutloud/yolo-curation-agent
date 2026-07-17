"""
Test: Orchestrator -> task("planning-agent") -> plan.md + class_budget.json

Flow being tested:

    User
       |
       v
    Orchestrator  (create_deep_agent, stripped to planning only)
       |
    task("planning-agent")
       |
       v
    Planning Agent
       |
       +-- plan.md
       +-- class_budget.json

The orchestrator is given ONLY the planning-agent as a subagent and is
instructed to delegate immediately (no HITL gates, no other subagents).
This exercises the real task() delegation path cleanly.

Exit codes:
    0  all assertions pass
    1  one or more assertions failed
    2  environment misconfigured
"""

import json
import os
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

LLM_MODEL = os.environ.get("PLANNING_TEST_MODEL", "gpt-oss:20b")

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
# Backend  --  same shared virtual filesystem as production
#   /workspace/ --> run_artifacts/ on real disk
# ------------------------------------------------------------------

from backends.project_backend import project_backend, RUN_ARTIFACTS_DIR

RUN_DIR = Path(RUN_ARTIFACTS_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"[ok] Backend  : /workspace/ -> {RUN_DIR}")

# ------------------------------------------------------------------
# Clean previous outputs so assertions are always fresh
# ------------------------------------------------------------------

EXPECTED_FILES = ["plan.md", "class_budget.json"]

for fname in EXPECTED_FILES:
    p = RUN_DIR / fname
    if p.exists():
        p.unlink()
        print(f"     Removed stale {fname}")

# ------------------------------------------------------------------
# Planning subagent  --  import from the real module
# ------------------------------------------------------------------

from subagents.planning import build_planning_agent

# Pass the same llm explicitly so deepagents uses it directly for the
# subagent instead of trying to re-derive it from the parent orchestrator.
# This is required when using ChatOllama with custom auth headers — deepagents
# cannot reconstruct those headers automatically during subagent propagation.
planning_agent = build_planning_agent(model=llm)

print(f"[ok] Subagent : {planning_agent['name']}")

# ------------------------------------------------------------------
# Minimal orchestrator  --  only knows about planning-agent
#
# No HITL gates, no sourcing/annotation/training subagents.
# Goal: purely test the User -> Orchestrator -> task() -> Agent path.
# ------------------------------------------------------------------

from deepagents import create_deep_agent

# Virtual backend path, not a host filesystem path — see
# backends/project_backend.py's "/skills/" route.
SKILLS_DIR = "/skills"

TEST_ORCHESTRATOR_PROMPT = """\
You are a test orchestrator. Your ONLY task is:

1. Call task("planning-agent") with the user's detection use-case details.
2. Wait for the planning-agent to finish and return its summary.
3. Reply with a short confirmation that includes the summary.

Do NOT call request_approval. Do NOT do anything else.
Write nothing to files yourself -- the planning-agent handles that.
"""

orchestrator = create_deep_agent(
    model=llm,
    system_prompt=TEST_ORCHESTRATOR_PROMPT,
    subagents=[planning_agent],   # only planning-agent is available
    tools=[],                     # no approval tool needed for this test
    skills=[SKILLS_DIR],
    backend=project_backend,
)

print("[ok] Orchestrator built (planning-agent only)")
print()

# ------------------------------------------------------------------
# User prompt  --  realistic PPE detection use-case
# ------------------------------------------------------------------

USER_PROMPT = """\
I need a YOLO model to detect the following classes on a Jetson Orin
running real-time video (target: >=15 FPS):

  - hard hat
  - safety vest
  - no hard hat (violation)
  - no safety vest (violation)

Please create a dataset plan: estimate images per class and propose
a train/val/test split.
"""

print("=" * 60)
print("USER PROMPT")
print("=" * 60)
print(USER_PROMPT.strip())
print("=" * 60)
print()
print("Running Orchestrator --> Planning Agent ...")
print()

# ------------------------------------------------------------------
# Invoke the orchestrator  (User -> Orchestrator -> task -> Agent)
# ------------------------------------------------------------------

response = orchestrator.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": USER_PROMPT,
            }
        ]
    }
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

plan_file   = RUN_DIR / "plan.md"
budget_file = RUN_DIR / "class_budget.json"

# 1. plan.md must exist and be non-empty
if not plan_file.exists():
    errors.append("FAIL: plan.md was not created under /workspace/")
elif len(plan_file.read_text(encoding="utf-8").strip()) == 0:
    errors.append("FAIL: plan.md exists but is empty")

# 2. class_budget.json must exist and be valid JSON
budget_data = None
if not budget_file.exists():
    errors.append("FAIL: class_budget.json was not created under /workspace/")
else:
    try:
        budget_data = json.loads(budget_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"FAIL: class_budget.json is not valid JSON -- {exc}")

# 3. class_budget.json should reference the expected classes
if budget_data is not None:
    content_str = json.dumps(budget_data).lower()
    for cls in ["hard hat", "safety vest"]:
        if cls not in content_str:
            errors.append(
                f"WARN: class_budget.json does not mention '{cls}' "
                f"-- planning agent may have missed a class"
            )

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------

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

print("-- plan.md " + "-" * 49)
print(plan_file.read_text(encoding="utf-8"))
print()
print("-- class_budget.json " + "-" * 39)
print(json.dumps(json.loads(budget_file.read_text(encoding="utf-8")), indent=2))
