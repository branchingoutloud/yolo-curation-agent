# The Sourcing Subagent — In Detail

## Where it sits in the pipeline

```
Orchestrator
    │
    ├─ [planning-agent]  → writes plan.md, class_budget.json
    ├─ [HITL Gate 1: approve plan]
    │
    ├─ [sourcing-agent] ◄──────── YOU ARE HERE
    │       │
    │       └─ writes sources.json
    │
    ├─ [annotation-agent]  (skipped for now)
    ├─ [dataset-agent]
    ...
```

The sourcing agent is the **first real data-touching step**. Everything before it is planning; everything after it depends on what it found.

---

## The Three Building Blocks

### 🔵 Block 1 — Context: What the agent receives

The agent doesn't receive structured function arguments. It receives a **natural language task prompt** from the orchestrator, which references files on the shared filesystem. The key handoffs are:

| What | File it comes from | Written by |
|---|---|---|
| Class list + image budget | `/workspace/class_budget.json` | `planning-agent` |
| Overall plan + rationale | `/workspace/plan.md` | `planning-agent` |
| Prior sourcing results (on re-runs) | `/workspace/sources.json` | Previous call to itself |

**Example first-call prompt** the orchestrator sends:
```
Search for datasets for classes: ["car", "truck", "bus"].
Per-class budget is in /workspace/class_budget.json.
Prefer CC0/CC-BY licensed, annotated in YOLO format.
```

**Example re-run prompt** (narrower — this is the key multi-round feature):
```
'bus' class only has 12 annotated images so far (target: 200).
Search specifically for bus detection datasets on Roboflow and Kaggle.
Append results to the existing /workspace/sources.json.
```

---

### 🔵 Block 2 — Execution: The three tools it uses

These are injected via `build_sourcing_agent(roboflow_tools, kaggle_tools, web_search_tools)` — all three come from `tools/mcp_clients.py`:

**Tool 1: Roboflow MCP** (`mcp.roboflow.com`)
- Requires `ROBOFLOW_API_KEY` env var — degrades to `[]` if missing (agent still boots)
- Used first — Roboflow Universe has annotated datasets, so it's the highest-value hit
- Can search, fork datasets, check annotation coverage

**Tool 2: Kaggle MCP** (`KAGGLE_MCP_URL` env var)
- Secondary source — good for academic/research datasets that may be unannotated
- Agent should note if annotations are missing so orchestrator can route to annotation-agent later

**Tool 3: `web_search` (Tavily)**
- Requires `TAVILY_API_KEY` — also optional/graceful
- Last resort for papers, niche image collections (e.g. Flickr, Google Open Images pointers)
- Returns URLs/references, not actual datasets — agent writes these as `status: "unannotated_collection"`

> All three degrade to empty `[]` if keys aren't set. The agent still starts — this is intentional (see the comment in `mcp_clients.py`).

**Tool 4 (auto-attached): Filesystem tools**
- `read_file`, `write_file`, `glob`, `grep` — injected by the `deepagents` harness automatically
- Used to read `class_budget.json`, read/write `sources.json`

---

### 🔵 Block 3 — Output: What it writes and says

**File written: `/workspace/sources.json`**

This is the contract between sourcing-agent and everything downstream (dataset-agent, annotation-agent, orchestrator). Proposed schema:

```json
[
  {
    "source": "roboflow_universe",
    "dataset_id": "traffic-detection/traffic-detection-4",
    "url": "https://universe.roboflow.com/...",
    "classes_covered": ["car", "truck"],
    "image_count": 1200,
    "annotation_format": "YOLO",
    "license": "CC BY 4.0",
    "annotation_coverage": { "car": 800, "truck": 400 },
    "quality_notes": "mixed lighting, some occlusion",
    "status": "available"
  },
  {
    "source": "kaggle",
    "dataset_id": "andrewmvd/car-plate-detection",
    "url": "https://kaggle.com/...",
    "classes_covered": ["bus"],
    "image_count": 300,
    "annotation_format": "Pascal VOC",
    "license": "CC0",
    "annotation_coverage": { "bus": 0 },
    "quality_notes": "unannotated raw images",
    "status": "needs_annotation"
  }
]
```

**⚠️ Critical rule:** The agent **reads `sources.json` first if it exists, then appends** — it never overwrites. This is explicitly stated in the system prompt and is what makes multi-round sourcing work correctly.

**Final message returned to orchestrator:**
```
Found 3 sources covering 'car' (1,850 images ✓ target: 500).
'truck' covered with 640 images ✓ target: 300.
'bus' unlabeled — 300 raw images on Kaggle (no annotations), 0 annotated found.
Recommend routing 'bus' to annotation-agent.
```

This short summary is what the orchestrator uses to decide next steps — it doesn't re-read `sources.json` for routing logic. The gap summary IS the routing signal.

---

## The Multi-Round Loop (the most important behavior to get right)

```
Round 1                   Round 2 (if orchestrator sees gap)
──────                    ──────────────────────────────────
task prompt:              task prompt:
"find datasets for        "bus still thin (12 images).
 car, truck, bus"          Search specifically for bus.
                           Append to sources.json."
       │                          │
       ▼                          ▼
Read class_budget.json    Read existing sources.json
Search all 3 sources      Search Roboflow+Kaggle for "bus"
Write sources.json        Append new entries, update counts
Return gap summary        Return updated gap summary
```

This is why the `description` field of the agent (what the orchestrator sees) explicitly says: _"Can be called multiple times — pass a specific class or coverage gap in the task prompt."_

---

## Testing the Agent in Isolation

Since engineers test each subagent before pushing, here's what verification looks like for sourcing specifically:

**Level 1 — No MCP (pure logic tests)**
- Mock `class_budget.json` → agent reads it correctly
- Run once → `sources.json` created with correct schema
- Run again → entries appended, no duplicates, no overwrite

**Level 2 — MCP integration**
- Real Roboflow call → at least 1 result for `"car detection"`
- Real Kaggle call → at least 1 result
- `sources.json` entries have all required fields

**Level 3 — Orchestrator integration**
- Feed gap summary back to orchestrator mock → does it correctly call sourcing again with a narrower brief?
- Verify the gap summary format is parseable enough for the orchestrator to route

---

## The One Thing That Can Break Everything

The `status` field in `sources.json` is critical. If sourcing doesn't flag `"needs_annotation"` correctly, the orchestrator has no signal to route to annotation-agent — it will try to use unannotated data as training data and fail silently. That's the highest-risk contract in the whole sourcing→dataset→training chain.
