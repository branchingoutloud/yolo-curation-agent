# Sourcing Subagent — Implementation Notes

Implementation companion to `sourcing-subagent-deep-dive.md` (design) covering
what is actually built, its contracts, edge cases, and how to test it in
isolation with mocked planning output.

## What exists

| Piece | File |
|---|---|
| SubAgent spec (prompt now embeds the full sources.json schema) | `subagents/sourcing.py` |
| sources.json contract: validation, dedupe check, coverage-vs-budget math | `tools/sources_schema.py` |
| Contract unit tests (no LLM, no keys) | `tests/test_sources_schema.py` |
| Mocked planning-agent output | `tests/fixtures/class_budget.json`, `tests/fixtures/plan.md` |
| Isolation harness (mock or real MCP tools) | `scripts/run_sourcing_isolated.py` |

## Inputs

The agent receives no structured arguments — only a natural-language task
prompt from the orchestrator plus files on the shared `/workspace/` filesystem
(persisted to `run_artifacts/` on disk):

| Input | Source | Notes |
|---|---|---|
| Task prompt | orchestrator (`task` call) | Full sweep, or a narrower brief naming one class/gap on re-runs |
| `/workspace/class_budget.json` | planning-agent (mocked by the fixture) | `classes[]` with `name`, `target_images`, `intra_class_variability`, `notes`; plus `use_case`, `deployment_target`, `split`, `total_target_images` |
| `/workspace/sources.json` | previous sourcing rounds | Read-first, append/update by `dataset_id`, never overwrite |
| Search tools | injected via `build_sourcing_agent(roboflow, kaggle, web)` | Any list may be empty (graceful degradation in `tools/mcp_clients.py`) |

## Outputs

**`/workspace/sources.json`** — a JSON array; per entry: `source`,
`dataset_id`, `url`, `classes_covered`, `image_count`,
`annotation_coverage` (per-class annotated counts, keys ⊆ `classes_covered`),
`annotation_format`, `license`, `status`, optional `quality_notes`.
`status` ∈ `available` | `needs_annotation` | `unannotated_collection` and is
the orchestrator's routing signal to annotation-agent.
`tools/sources_schema.py` is the executable definition of this contract.

**Final message** — a short per-class gap summary (annotated found vs target,
what still needs annotation or more sourcing). No file contents or raw search
dumps; the file is the record, the reply is the routing signal.

## Edge cases handled

- **No tools configured** — MCP loaders degrade to `[]`; the harness refuses
  to run with zero search tools; the prompt tells the agent to say so and
  write nothing rather than invent results.
- **Re-run/append** — prompt mandates read-first + update-by-`dataset_id`;
  `find_duplicate_dataset_ids()` catches violations after the run.
- **Unannotated data mislabeled as available** — the highest-risk failure.
  Prompt pins the three-value `status` enum; `validate_sources()` rejects
  anything else; `coverage_summary()` counts only `available` sources as
  annotated (everything else is `raw`).
- **Fabrication** — prompt forbids entries not backed by a tool result.
- **Partial coverage** — `coverage_summary()` reports `annotated`, `raw`,
  `gap`, `met` per class so the orchestrator can re-brief precisely.

## Testing

**Level 1a — contract only (no keys):**

```bash
.venv/bin/python -m pytest tests/ -q
```

**Level 1b — behavioral, mocked search tools (needs `ANTHROPIC_API_KEY`):**

```bash
python scripts/run_sourcing_isolated.py --mock --fresh    # round 1
python scripts/run_sourcing_isolated.py --mock --round2   # append round
```

The mocks are canned so round 1 finds car/truck annotated but bus only raw
(`needs_annotation` path), and a targeted "bus" query surfaces an annotated
bus set (append/update path). The harness validates `sources.json` against
the contract and prints coverage vs budget; exit code 0 = PASS.

**Level 2 — real MCP (needs `ROBOFLOW_API_KEY` and/or `KAGGLE_MCP_URL`,
optional `TAVILY_API_KEY`):**

```bash
python scripts/run_sourcing_isolated.py --fresh
```

**Level 3 — orchestrator integration:** run `langgraph dev` and drive the full
agent; not covered by this harness.

## Other notes

- `SOURCING_MODEL` env var overrides the model for isolated runs (falls back
  to `ORCHESTRATOR_MODEL`).
- The harness runs the exact SubAgent spec (same prompt/tools/backend) as a
  top-level agent — mirroring what `SubAgentMiddleware` does, minus the
  orchestrator.
- Roboflow/Kaggle hits won't always report per-class annotation counts
  directly; the agent may need a follow-up detail/health-check call per
  candidate dataset before it can fill `annotation_coverage` honestly.
