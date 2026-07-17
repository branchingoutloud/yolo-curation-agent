# Autonomous YOLO Curation & Training Agent — Architecture

A `deepagents`-native system that takes a user's detection use case, sources and curates
training data, trains a YOLO model, evaluates it, and proposes an iteration — demonstrating
subagents, adaptive orchestration, context management, sandboxed code execution, MCP tools,
Agent Skills, and human-in-the-loop approval, all as first-class `deepagents` features
rather than bolted-on custom code.

**The orchestrator is not given a fixed script.** It's given a goal, a roster of subagents,
and judgment calls to make — it decides how many rounds of sourcing are enough, whether a
weak class needs more data or just re-annotation, and when a checkpoint is actually ready
to show the user. See §3.

Framework: [`langchain-ai/deepagents`](https://github.com/langchain-ai/deepagents) (Python).
Models are left configurable — pass any `init_chat_model`-compatible string per agent/subagent.

---

## 1. Why `deepagents` fits this project

| Deep Agents capability | How this project uses it |
|---|---|
| `task` tool / subagents | Each pipeline stage (planning, sourcing, annotation, training, eval) is an isolated subagent so the orchestrator's context never sees raw search dumps or training logs |
| Agent-driven sequencing ("trust the LLM") | The orchestrator decides *when* and *how often* to call each subagent based on what it reads back — re-sourcing, re-annotating, or retraining are its own judgment calls, not hardcoded branches |
| Virtual filesystem + pluggable backends | The plan, sourced-data manifest, dataset spec, and metrics all live as files, not chat turns — the demo's core "context management" beat |
| Sandboxes (`execute` tool) | Zero-shot annotation and `ultralytics` training run as real shell/GPU execution, isolated from the orchestrator's process |
| `interrupt_on` (HITL middleware) | Three approval gates pause the graph and wait for the user, using LangGraph's native interrupt/resume rather than custom prompting |
| MCP tool support | Roboflow and Kaggle MCP servers are passed straight into `tools=` / a subagent's `tools=` list |
| Agent Skills (`SkillsMiddleware`) | On-demand YOLO/CV domain knowledge (model sizing, dataset curation heuristics, eval diagnosis) loads only when a subagent's task needs it |
| Context summarization | Long subagent threads (e.g. a 50-epoch training run) auto-summarize instead of ballooning the orchestrator's token count |

---

## 2. Architecture at a glance

```
User
 │
 ▼
Orchestrator (create_deep_agent)
 │  system_prompt: interview → plan → delegate → gate → delegate → gate → delegate → gate
 │
 ├─ task("planning-agent")        → writes plan.md, class_budget.json
 ├─ [HITL gate 1: approve plan]
 ├─ task("sourcing-agent")        → writes sources.json          (Roboflow MCP, Kaggle MCP, web_search)
 ├─ task("annotation-agent")      → writes annotated/ manifest    (sandbox: zero-shot model)
 ├─ task("dataset-agent")         → writes dataset/data.yaml
 ├─ [HITL gate 2: confirm model size]
 ├─ task("training-agent")        → writes runs/train/metrics.json (Modal sandbox: ultralytics)
 ├─ task("eval-agent")            → writes eval_report.md, weak_classes.json
 ├─ [HITL gate 3: approve iteration 2]
 └─ loop back to sourcing-agent for iteration 2
```

Everything left of an arrow is a `task()` call the orchestrator makes; everything after
`writes` is a file in the shared virtual filesystem, which is how subagents hand off
results without polluting each other's — or the orchestrator's — context window.

---

## 3. Orchestrator (main agent) — goal-driven, not scripted

**Design change from the first draft:** the orchestrator is not given a numbered
script to execute, or even a manually-written roster of subagents. It's given a
**goal, a set of judgment calls it's expected to make, and a `write_todos` tool
to plan for itself** — the subagent roster itself comes for free from the
`subagents=` list passed to `create_deep_agent` (`deepagents`' `SubAgentMiddleware`
injects each one's `name`/`description` into the `task` tool automatically, so
the prompt doesn't need to repeat it). This is the actual "trust the LLM" model
`deepagents` is built around — the harness gives the agent planning and
delegation tools; the *sequencing* is a decision the model makes at runtime,
not a hardcoded workflow.

Concretely, this means:
- The orchestrator decides **how many rounds of sourcing** are needed — if
  `sourcing-agent` returns too few images for a class, the orchestrator can call it
  again with a narrower brief, rather than a fixed one-shot search.
- After training, the orchestrator **reads `eval_report.md` itself** and reasons
  about *why* a class underperformed — too few images, high visual variability,
  label noise, a class the zero-shot annotator struggled with — and chooses the
  matching next action: call `sourcing-agent` for more raw images, call
  `annotation-agent` to re-label existing ones, or just call `training-agent` again
  with adjusted hyperparameters (no new data at all).
- The three HITL gates are still the right places to pause a human, but the
  orchestrator reaches them **because it decided the plan/model choice/iteration
  plan is ready to present** — not because a step counter said so. It can loop
  sourcing ↔ annotation ↔ planning multiple times before ever reaching gate 1.

```python
from deepagents import create_deep_agent

ORCHESTRATOR_PROMPT = """
You are the orchestrator for a YOLO data-curation and training agent. Your goal:
take a user's detection use case and produce a trained, evaluated YOLO model,
improving it over as many iterations as the data and eval results warrant.

You have no data-sourcing, annotation, or training abilities yourself — you only
plan, delegate to subagents via `task`, read their output files, reason about
whether the result is good enough, and either delegate further or ask the user
to weigh in via `request_approval`.

Use `write_todos` to track your own plan and update it as you learn more —
do not assume the pipeline only runs once. After every eval-agent result,
decide for yourself whether another sourcing/annotation/training round is
warranted, and say why, before proposing it to the user.

Call `request_approval` (and only then) at these three natural checkpoints:
(1) once you have a sourcing/image-budget plan you're confident in and before
    any data is actually pulled or forked into a workspace,
(2) once you've proposed a specific YOLO model size and before training starts,
(3) once you have eval results and a concrete, reasoned iteration plan.
Do not call `request_approval` at any other time, and do not skip these three.

Never fabricate metrics, dataset stats, or file contents — always read them
from the filesystem tools first.
"""

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",  # any init_chat_model string works
    system_prompt=ORCHESTRATOR_PROMPT,
    subagents=[planning_agent, sourcing_agent, annotation_agent,
               dataset_agent, training_agent, eval_agent],
    tools=[request_approval],  # see §9 — the only non-delegation tool it needs
    interrupt_on={
        "request_approval": {
            "allowed_decisions": ["approve", "edit", "reject"],
        },
    },
    skills=["./skills"],  # see §6
    backend=project_backend,  # see §8
)
```

Note what's deliberately *not* in the prompt: a roster of subagents and what
each one does. `deepagents`' `SubAgentMiddleware` already builds that into the
`task` tool's own description at the harness level, straight from each
`SubAgent`'s `name`/`description` fields (§4) — restating it in the system
prompt would just be duplicated context. The prompt above only needs to carry
what the harness *can't* infer on its own: the goal, the judgment calls, and
exactly where the three gates belong.

This is also why gating the `task` tool itself (as opposed to a dedicated
`request_approval` tool) would be the wrong choice here even mechanically: if the
orchestrator is going to call `sourcing-agent` an unknown number of times, an
interrupt on every `task` call would pause the graph on every one of those rounds,
not just the three moments that actually need a human.

---

## 4. Subagents

| Name | Job | Key tools | Sandbox? | Writes |
|---|---|---|---|---|
| `planning-agent` | Estimate classes, per-class image budget (low intra-class variance → fewer images needed), propose dataset size/split | filesystem only | No | `plan.md`, `class_budget.json` |
| `sourcing-agent` | Search Roboflow Universe, Kaggle, and the web for matching datasets/papers; fork or download candidates | Roboflow MCP, Kaggle MCP, `web_search` | No (network calls only) | `sources.json` |
| `annotation-agent` | For classes with no labeled data: run a zero-shot open-vocab detector to pre-label, push predictions to Roboflow for human verification | `execute` (sandbox), Roboflow MCP | **Yes** — needs GPU/CPU for inference | `annotation_manifest.json` |
| `dataset-agent` | Merge sources, dedupe, split train/val/test, emit `data.yaml` | filesystem, `execute` | Optional (light) | `dataset/data.yaml` |
| `training-agent` | Select YOLO variant per deployment target, launch `ultralytics` training, stream periodic metric summaries | `execute` (Modal sandbox) | **Yes** — GPU required | `runs/train/metrics.json`, `runs/train/status.md` |
| `eval-agent` | Parse results, build confusion matrix via `supervision`, identify weak classes, draft iteration-2 data plan | `execute` (same sandbox) | Yes (reuses training sandbox) | `eval_report.md`, `weak_classes.json` |

```python
from deepagents import SubAgent

planning_agent: SubAgent = {
    "name": "planning-agent",
    "description": (
        "Given target classes, use case, and deployment target, estimates "
        "images-per-class needed and proposes a dataset size/split. Use "
        "this before any data is sourced."
    ),
    "system_prompt": (
        "You are a dataset-planning specialist. Lower intra-class visual "
        "variability (e.g. a single fixed logo) needs far fewer images than "
        "high-variability classes (e.g. 'pedestrian' across poses/lighting). "
        "Write your reasoning and final numbers to plan.md and "
        "class_budget.json. Return only a one-paragraph summary."
    ),
    "tools": [],  # filesystem tools are auto-attached by the harness
}

sourcing_agent: SubAgent = {
    "name": "sourcing-agent",
    "description": (
        "Searches Roboflow Universe, Kaggle, and the web for datasets matching "
        "the class budget. Can be called multiple times — pass a specific "
        "class or coverage gap in the task prompt for a narrower, targeted "
        "follow-up search rather than a full re-run."
    ),
    "system_prompt": (
        "Search for public datasets and annotated data matching the classes "
        "you were asked about (all of them, or a specific subset named in your "
        "task). Prefer already-annotated sources. If sources.json already "
        "exists, read it first and append/update rather than overwrite. Write "
        "every candidate (source, license, image count, annotation coverage) "
        "to sources.json. Return only a short summary of what was found and "
        "what remains unlabeled or thin."
    ),
    "tools": [*roboflow_mcp_tools, *kaggle_mcp_tools],
}

annotation_agent: SubAgent = {
    "name": "annotation-agent",
    "description": "Zero-shot pre-labels images for classes with no annotated data found.",
    "system_prompt": (
        "For each class flagged unlabeled in sources.json, run the zero-shot "
        "detector via execute() over the candidate images, push predictions "
        "into the Roboflow project as draft annotations for human review, "
        "and record results in annotation_manifest.json."
    ),
    "tools": [*roboflow_mcp_tools],
    "backend": annotation_sandbox_backend,  # per-subagent sandbox override
}

training_agent: SubAgent = {
    "name": "training-agent",
    "description": "Selects a YOLO model size and runs ultralytics training in a GPU sandbox.",
    "system_prompt": (
        "Read dataset/data.yaml and the confirmed model size from "
        "model_choice.json. Launch training via execute(). Every few epochs, "
        "append a short status line to runs/train/status.md — never print "
        "raw ultralytics logs to your final message. On completion write "
        "runs/train/metrics.json."
    ),
    "tools": [],
    "backend": training_sandbox_backend,  # Modal GPU sandbox
}

eval_agent: SubAgent = {
    "name": "eval-agent",
    "description": (
        "Analyzes a completed training run: confusion matrix, per-class "
        "metrics, and a best-guess diagnosis for each weak class (too few "
        "images, high visual variability, likely label noise, or a class the "
        "zero-shot annotator struggled with). Does NOT decide the next "
        "action — that's the orchestrator's call based on this diagnosis."
    ),
    "system_prompt": (
        "Read runs/train/metrics.json and the trained weights. Use the "
        "`supervision` library via execute() to build a confusion matrix and "
        "sample failure crops. For each underperforming class, inspect its "
        "sample count, source annotation quality, and failure crops, and "
        "record a specific likely cause — not just the metric. Write "
        "eval_report.md (human-readable) and weak_classes.json (structured: "
        "class name, metric, sample count, likely cause) so the orchestrator "
        "can decide whether to re-source, re-annotate, or just retune."
    ),
    "tools": [],
    "backend": training_sandbox_backend,  # reuse — weights are already there
}
```

---

## 5. Tools & MCP servers

| Tool | Type | Used by |
|---|---|---|
| Roboflow MCP (`mcp.roboflow.com`) | MCP server — project mgmt, dataset upload, auto-label, fork Universe datasets, dataset health check | `sourcing-agent`, `annotation-agent` |
| Kaggle MCP (official or community) | MCP server — dataset/notebook search & download | `sourcing-agent` |
| `web_search` | Built-in LangChain tool | `sourcing-agent` (papers, niche public data pointers) |
| `zero_shot_annotate` | Custom tool wrapping Grounding DINO / YOLO-World inference | `annotation-agent` (runs inside its sandbox via `execute`) |
| `execute` | Built-in Deep Agents sandbox tool | `annotation-agent`, `dataset-agent`, `training-agent`, `eval-agent` |
| filesystem tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) | Built-in | All agents |

Attaching MCP servers:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient({
    "roboflow": {
        "url": "https://mcp.roboflow.com",
        "transport": "streamable_http",
        "headers": {"Authorization": f"Bearer {ROBOFLOW_API_KEY}"},
    },
    "kaggle": {
        "url": KAGGLE_MCP_URL,
        "transport": "streamable_http",
    },
})
roboflow_mcp_tools = await mcp_client.get_tools(server_name="roboflow")
kaggle_mcp_tools = await mcp_client.get_tools(server_name="kaggle")
```

A custom tool is just a decorated Python function — no MCP needed for the
zero-shot annotator since it runs inside the sandbox rather than as a
separate service:

```python
from langchain_core.tools import tool

@tool
def zero_shot_annotate(image_dir: str, classes: list[str], out_dir: str) -> str:
    """Run a zero-shot open-vocabulary detector over image_dir for the given
    classes, writing YOLO-format label files to out_dir. Returns a short
    summary (image count, per-class detection count)."""
    ...
```

---

## 6. Skills — giving agents CV/YOLO domain knowledge

`deepagents` supports the same **Agent Skills** open format Claude Code uses:
a `skills/` directory of folders, each with a `SKILL.md` (YAML frontmatter +
markdown instructions, optionally with `scripts/`, `references/`, `assets/`).
At startup only each skill's `name`/`description` is injected into the system
prompt (~100 tokens each); the full body loads only when a subagent's task
actually matches it. This is on-demand domain expertise, not upfront bloat —
exactly what a curation/training pipeline needs, since the planning, dataset,
and eval subagents each need different specialist knowledge at different times.

```python
agent = create_deep_agent(
    ...,
    skills=["./skills"],   # scanned for */SKILL.md
)

# a subagent can be scoped to only the skills it needs:
eval_agent: SubAgent = {
    ...,
    "skills": ["./skills/cv-eval-and-iteration"],
}
```

### What's already out there (researched, not assumed)

- **`voxel51/fiftyone-skills`** (GitHub, maintained by the FiftyOne team) has a
  genuinely solid `fiftyone-dataset-curation` skill: schema inspection, class-
  distribution/imbalance audits, near-duplicate detection, embedding-based gap
  detection, and train/val/test split logic — all directly reusable if you wire
  in FiftyOne as your dataset backend, and a good reference even if you don't.
- Community CV skills exist (an `alirezarezvani/senior-computer-vision` skill
  circulating on OpenClaw skill mirrors covers dataset format conversion,
  augmentation config generation, and export/quantization for edge deployment —
  useful *reading* for what a good SKILL.md in this space looks like) — but
  provenance on community mirrors is unverified, treat them as inspiration to
  adapt, not as-is dependencies for a judged hackathon submission.
- Marketplace listings claiming to be "YOLO integration" skills (e.g. on
  LobeHub) were mostly template placeholders with no real instructional
  content when inspected — not worth pulling in as-is.

**Given that, the highest-leverage move for 24-36 hours is writing 2-3 tight,
project-specific skills yourself** — they're small (a few hundred lines of
markdown each), directly demonstrate the Skills capability to judges, and
encode exactly the domain judgment your subagents need:

| Skill | Used by | Encodes |
|---|---|---|
| `yolo-model-selection` | `planning-agent`, `training-agent` | Model size vs. deployment-target matrix, default hyperparameters per size, when to prefer YOLO11 vs YOLOv8 |
| `cv-dataset-curation` | `sourcing-agent`, `dataset-agent` | Images-per-class heuristics by intra-class variability, class imbalance thresholds, split ratios, dedup/quality checks (borrow directly from `fiftyone-dataset-curation`'s approach) |
| `cv-eval-and-iteration` | `eval-agent` | How to read mAP50/mAP50-95, precision/recall trade-offs, confusion-matrix patterns that indicate *too little data* vs *label noise* vs *class confusability*, and what each pattern implies for the next round |

Example — `skills/yolo-model-selection/SKILL.md`:

```markdown
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
   one size up from the target-implied default — small models under-fit
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
recommending one size up before recommending more data — small models can be
data-starved in a way that looks like a data problem but is a capacity problem.
```

The other two skills follow the same shape: a short decision table, a rule for
when to apply it, and an explicit note on what changes the recommendation on a
second pass — since these subagents may be invoked more than once per run (§3).

---

## 7. Filesystem backend & context management

All pipeline state lives in a shared virtual filesystem, not in message history.
This is the concrete mechanism behind the "context management" demo point:

```python
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

project_backend = CompositeBackend(
    default=StateBackend(),          # ephemeral scratch space
    routes={
        "/workspace/": FilesystemBackend(root_dir="./run_artifacts"),
        # persists plan.md, sources.json, dataset/, runs/, eval_report.md
        # to real disk so a judge can literally open the files after the demo
    },
)
```

Every subagent is instructed to write its full output to a file under
`/workspace/` and return only a short summary as its final message — the
`task` tool already hides intermediate subagent steps from the orchestrator
by design, so this just extends the same discipline to *artifacts*, not only
reasoning steps.

Demo beat: open LangSmith and show the orchestrator's trace staying flat
(a handful of short messages) while the `training-agent`'s own trace is long —
proof the isolation is real, not just described.

---

## 8. Sandboxes

Two different sandbox needs, kept separate:

- **Annotation sandbox** — CPU or small GPU, short-lived, just needs to run a
  zero-shot model over a batch of images.
- **Training sandbox** — GPU required, longer-lived (spans the whole training
  run), reused by `eval-agent` afterward so weights/results are already local.

```python
from deepagents.backends.sandbox import ModalSandbox  # or DaytonaSandbox

training_sandbox_backend = ModalSandbox(
    image="ultralytics/ultralytics:latest-python",
    gpu="A10G",
    timeout=60 * 45,
)

annotation_sandbox_backend = ModalSandbox(
    image="python:3.11-slim",  # + grounding-dino / yolo-world deps
    gpu="T4",
    timeout=60 * 15,
)
```

This mirrors the pattern used in `deepagents`' own `nvidia_deep_agent` example,
which integrates Modal for GPU-accelerated subagent execution — a good talking
point for judges ("we're using the same sandbox pattern as LangChain's own
reference example, not a bespoke hack").

---

## 9. Human-in-the-loop gates

Three gates, per the earlier design decision. Since `interrupt_on` keys on
tool name and every delegation goes through `task`, the cleanest way to get
*stage-specific* gates without pausing on every subagent call is to gate a
small, explicit "checkpoint" tool that only the orchestrator calls at those
three points (rather than gating `task` itself):

```python
from langchain_core.tools import tool

@tool
def request_approval(stage: str, summary: str) -> str:
    """Present a plan/decision to the user and pause for approval before
    proceeding. Call this exactly at: (1) after planning, before sourcing;
    (2) after proposing a model size, before training; (3) after eval,
    before an iteration-2 sourcing pass."""
    return f"Approval requested for {stage}: {summary}"

agent = create_deep_agent(
    ...,
    tools=[request_approval],
    interrupt_on={
        "request_approval": {"allowed_decisions": ["approve", "edit", "reject"]},
    },
)
```

The orchestrator's system prompt (§3) already instructs it to "STOP for
approval" at exactly those three points — `request_approval` gives that
instruction a concrete tool to call, and LangGraph's interrupt/resume handles
the actual pause.

---

## 10. Process flow — an adaptive loop, not a fixed script

The diagram shown earlier is still the right *shape* for the demo narrative,
but the orchestrator isn't walking through it as eight fixed steps — each box
can be visited more than once, in an order the orchestrator itself decides
based on what it reads back from the filesystem:

1. **User intake** — orchestrator interviews for classes, use case, deployment target.
2. **Planning** — `planning-agent` proposes a class budget. The orchestrator may
   call it again later if sourced coverage doesn't match the original estimate.
3. **Gate 1** — reached when the orchestrator judges the plan is solid, not on a step count.
4. **Sourcing + annotation** — `sourcing-agent` and `annotation-agent`, possibly
   several rounds, possibly interleaved, targeting whichever class is thinnest.
5. **Dataset assembly** — `dataset-agent` merges whatever has been sourced so far.
6. **Gate 2** — model size proposal, reached once the orchestrator has a dataset it trusts.
7. **Training** — `training-agent` in the GPU sandbox.
8. **Evaluation** — `eval-agent` returns a diagnosis per weak class, not just a metric.
9. **Orchestrator reasoning** — for each weak class, it decides: more raw images
   (→ step 4, sourcing), re-annotation (→ step 4, annotation only), or a
   hyperparameter change with no new data (→ straight back to step 7).
10. **Gate 3** — reached once that reasoning is complete, presenting the concrete
    plan and *why*, before looping back to whichever earlier step it chose.

This is also why `eval-agent`'s output includes a likely cause per class
(§4) — the orchestrator needs that to route correctly to step 4 vs. step 7
rather than always defaulting to "get more data."

---

## 11. Suggested repo structure

```
yolo-deep-agent/
├── agent.py                  # create_deep_agent() wiring, orchestrator prompt
├── subagents/
│   ├── planning.py
│   ├── sourcing.py
│   ├── annotation.py
│   ├── dataset.py
│   ├── training.py
│   └── eval.py
├── skills/
│   ├── yolo-model-selection/SKILL.md
│   ├── cv-dataset-curation/SKILL.md
│   └── cv-eval-and-iteration/SKILL.md
├── tools/
│   ├── mcp_clients.py        # Roboflow/Kaggle MCP setup
│   ├── zero_shot_annotate.py
│   └── approval.py           # request_approval tool
├── backends/
│   ├── project_backend.py    # CompositeBackend / FilesystemBackend
│   └── sandboxes.py          # ModalSandbox configs
├── run_artifacts/            # persisted plan.md, sources.json, dataset/, runs/
└── README.md
```

---

## 12. 24–36hr build order (solo)

| Time | Milestone |
|---|---|
| 0–3h | Orchestrator skeleton + `planning-agent`, no real data yet — verify `task` delegation and Gate 1 pause/resume |
| 3–8h | `sourcing-agent` with real Roboflow + Kaggle MCP calls; hardcode one demo use case as a fallback |
| 8–13h | `annotation-agent` with zero-shot model in a sandbox; verify predictions land in Roboflow for review |
| 13–17h | `dataset-agent` + Gate 2 + `training-agent` on Modal, get one real end-to-end training run working (cache it) |
| 17–22h | `eval-agent` + Gate 3 + iteration loop; wire LangSmith tracing for the context-management demo beat |
| 22–28h | Rehearse the full demo script twice at reduced epoch count; fix the rough edges |
| 28–36h | Buffer / polish / slides |

---

## 13. Risks & fallbacks

- **Live GPU spin-up latency during judging** → pre-run once, cache the dataset and a checkpoint locally; live demo trains a small held-out class at ~10-15 epochs only.
- **Roboflow/Kaggle rate limits or auth hiccups live** → keep a cached `sources.json` from a dry run as a fallback path the orchestrator can read if the MCP call fails.
- **Zero-shot model too slow/heavy for a hackathon laptop** → keep it swappable behind `zero_shot_annotate`; YOLO-World is lighter than Grounding DINO if the sandbox GPU is constrained.
