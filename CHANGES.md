# Changes made — planning & training subagent setup

Context: the repo had a full scaffold matching `yolo-deep-agent-architecture (1).md`,
but `agent.py` could not actually be imported, and two more bugs were silently
swallowed by the original backend setup. This document explains what was
found, what was changed, and why — organized by root cause, then by file.

## 1. `agent.py` could not import at all

**Symptom:** `python -c "import agent"` raised
`ImportError: cannot import name 'ModalSandbox' from 'deepagents.backends.sandbox'`.

**Root cause:** `backends/sandboxes.py` imported `ModalSandbox` from
`deepagents.backends.sandbox`. That class does not exist in the installed
deepagents version (0.6.12) — `deepagents/backends/sandbox.py` only defines
the abstract `BaseSandbox`. The architecture doc's §8 code sample describing
`ModalSandbox` was aspirational, not something this deepagents version ships.

**Fix — `backends/sandboxes.py` (rewritten):**
Replaced the Modal import with a `build_sandbox_backend()` selector, chosen
via a new `SANDBOX_BACKEND` env var:
- `"local"` (default) → `deepagents.backends.LocalShellBackend`, which runs
  `execute()` directly on this machine. Zero setup, works immediately since
  `ultralytics`/`torch` are already in this repo's venv.
- `"docker"` → the new `DockerSandbox` (see below), for real process
  isolation.

## 2. Per-subagent `"backend"` overrides do nothing in this deepagents version

**Symptom:** none visible — this is a silent no-op, which is why it hadn't
been noticed.

**Root cause:** `subagents/annotation.py`, `subagents/training.py`, and
`subagents/eval.py` each set a `"backend": sandbox_backend` key on their
`SubAgent` dict, following the architecture doc's §4/§8 "per-subagent
sandbox override" design. I checked `deepagents.middleware.subagents`
(`create_sub_agent`, `_build_task_tool`, `SubAgentMiddleware`) and
`deepagents.graph` directly — none of them read a `"backend"` key from a
`SubAgent` spec anywhere. `SubAgent` is a `TypedDict`, so Python doesn't
error on the extra key; it's just discarded.

**Consequence:** there is exactly **one** sandbox for the whole graph — the
one passed to `create_deep_agent(backend=...)` — not one per subagent as the
architecture doc describes.

**Fix:**
- `subagents/training.py`, `subagents/annotation.py`, `subagents/eval.py` —
  removed the dead `"backend"` field and the now-unused `sandbox_backend`
  parameter from each `build_*_agent()` function. Left a short comment on
  each pointing to the explanation (in `training.py`'s docstring) so nobody
  re-adds it expecting it to work.
- The single real sandbox is now configured once, in
  `backends/project_backend.py` (see §3).

## 3. `execute()` never actually worked for any subagent

**Symptom:** would have surfaced as `NotImplementedError: Default backend
doesn't support command execution` the first time any subagent called
`execute()`.

**Root cause:** `CompositeBackend.execute()` (installed deepagents source,
`deepagents/backends/composite.py`) **always** delegates to `self.default`,
never to a routed backend — routing only applies to file operations
(`read`/`write`/`edit`/`ls`/`glob`/`grep`), not to `execute()`. The original
`project_backend.py` had:
```python
CompositeBackend(default=StateBackend(), routes={"/workspace/": FilesystemBackend(...)})
```
`StateBackend` is not a `SandboxBackendProtocol` implementation, so
`execute()` had no working backend regardless of what was routed under
`/workspace/`.

**Fix — `backends/project_backend.py` (rewritten):** `default` is now the
same sandbox-capable backend that's routed for `/workspace/`:
```python
project_backend = CompositeBackend(
    default=sandbox_backend,
    routes={
        "/workspace/": sandbox_backend,
        "/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True),
    },
)
```
Effects:
- Reads/writes under `/workspace/` behave exactly as before (prefix
  stripped, persisted to `RUN_ARTIFACTS_DIR`) — no change to existing
  subagent prompts or file layout.
- `execute()` now actually runs, on the host (`local`) or in Docker
  (`docker`), instead of raising `NotImplementedError`.
- Paths outside `/workspace/` now also land under `RUN_ARTIFACTS_DIR`
  instead of vanishing into `StateBackend`'s ephemeral scratch space —
  harmless here since every subagent is instructed to write under
  `/workspace/` anyway.

## 4. Orchestrator model was a bare string — breaks Ollama Cloud auth

**Symptom:** would have failed the first time the orchestrator (or any
subagent without its own `model=`) tried to call the model, since it would
silently target `http://localhost:11434` with no auth instead of Ollama
Cloud.

**Root cause:** `agent.py` passed
`model=os.environ.get("ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5")` —
a bare string — to `create_deep_agent`. For providers like `anthropic:`/
`openai:`, `init_chat_model` reads auth from standard env vars and a string
is fine. `ollama:` is different: Ollama Cloud needs a custom `base_url` +
an `Authorization: Bearer <key>` header, which `init_chat_model("ollama:...")`
has no way to supply — it just builds a `ChatOllama` pointed at the local
default. Additionally, deepagents resolves each subagent's model as
`spec.get("model", model)` — a subagent with no `model` override inherits
whatever was passed to `create_deep_agent`. If that's a *string*, deepagents
re-runs `init_chat_model()` per subagent and drops the custom headers again,
every time.

**Fix — `tools/models.py` (new):** `build_default_model()` reads
`ORCHESTRATOR_MODEL`; if the provider prefix is `ollama`, it constructs a
real `ChatOllama` object with the base URL + auth header (matching the
pattern already used ad hoc in `test_planning_agent.py`/`ollama_test.py`);
otherwise it delegates to `init_chat_model`. Passing an **object** (not a
string) as `create_deep_agent(model=...)` means every subagent that doesn't
set its own `model` inherits that exact authenticated instance — no
re-resolution, no dropped headers.

**Fix — `agent.py`:** now calls `model = build_default_model()` and passes
the object through.

**Fix — `subagents/training.py`:** added the same `model: str | BaseChatModel
| None = None` override parameter that `subagents/planning.py` already had,
for symmetry and for standalone-orchestrator tests that don't go through
`agent.py`'s model resolution at all.

## 5. `skills=[...]` was passing a real filesystem path, not a backend path

**Symptom:** `ValueError: Path:C:\...\skills outside root directory:
C:\...\run_artifacts`, raised from `SkillsMiddleware.before_agent` the first
time an agent turn ran (only surfaced *after* fixing #3 — see below).

**Root cause:** `agent.py` (and both test scripts) set
`SKILLS_DIR = str(Path(__file__).resolve().parent / "skills")` — an absolute
host filesystem path — and passed it as `skills=[SKILLS_DIR]`. But per
`deepagents.middleware.skills`'s own module docstring: "Sources point to
skill directories **in the backend**" — these are meant to be virtual
backend paths (like `/workspace/...`), resolved through the same backend as
everything else, not raw filesystem paths. `SkillsMiddleware` calls
`backend.ls(source_path)` directly with whatever string it's given.

This was **already broken silently** before any of today's changes: with the
old `default=StateBackend()`, `StateBackend.ls()` doesn't validate the path
against a root directory — it just found nothing under that unfamiliar key
and returned an empty skill list. That means **zero skills were ever
actually loading** in this project, despite the architecture doc's §6
emphasis on them. Fixing #3 (making `default` a real, root-validating
`FilesystemBackend`/`LocalShellBackend`) turned that silent failure into a
loud one, which is how it was caught.

**Fix:**
- `backends/project_backend.py` — added a `"/skills/"` route to a
  `FilesystemBackend` rooted at the repo's actual `skills/` directory.
- `agent.py`, `test_planning_agent.py`, `test_training_agent.py` —
  `SKILLS_DIR` changed from an absolute host path to the virtual path
  `"/skills"`.

## New files

- **`backends/docker_sandbox.py`** — `DockerSandbox(FilesystemBackend,
  SandboxBackendProtocol)`. File I/O is inherited from `FilesystemBackend`
  and operates on the host path directly (works because the container
  bind-mounts that same directory — see docker-compose.yml). Only
  `execute()`/`aexecute()`/`id` are overridden, running commands via
  `docker exec <container> sh -c "<command>"`. Mirrors the shape of
  deepagents' own `LocalShellBackend` (extend `FilesystemBackend`, add
  `execute()`), since no shipped class already does this.
- **`docker/Dockerfile`** — `python:3.11-slim` + `ultralytics` + `supervision`,
  CPU-only (this machine has no NVIDIA GPU — confirmed via `nvidia-smi`
  not being present). Stays alive via `CMD ["sleep", "infinity"]` so
  `docker exec` can be called repeatedly across a whole training run instead
  of paying container-start latency per command.
- **`docker/docker-compose.yml`** — builds the image, bind-mounts
  `../run_artifacts` to `/workspace`, names the container
  `yolo-training-sandbox` (matches `DockerSandbox`'s default). GPU
  passthrough is commented in for a machine that has one.
- **`tools/models.py`** — see §4.
- **`test_training_agent.py`** — mirrors `test_planning_agent.py`'s
  structure: builds an isolated orchestrator with only `training-agent` as a
  subagent (no HITL gates, no other subagents), so it cleanly exercises
  `User → Orchestrator → task() → training-agent → execute()`. Since
  `dataset-agent` (which would normally produce `dataset/data.yaml`) is out
  of scope, the test stages its own fixture:
  - a tiny synthetic 8-image YOLO-format dataset (6 train / 2 val, one class
    `widget`, generated with Pillow — no network dependency), and
  - `model_choice.json` requesting `model: yolo11n.yaml` (architecture only,
    random init — **not** a `.pt` checkpoint) for 1 epoch at `imgsz=64`.

  Using `yolo11n.yaml` instead of pretrained weights means the whole test is
  offline-capable (no weight download), which matters for a repeatable CI/
  judge-machine smoke test — only the Ollama Cloud call itself needs
  network access.

  Assertions: `runs/train/status.md` and `runs/train/metrics.json` exist and
  are non-empty/valid JSON (hard fail), a soft warning if `metrics.json`
  doesn't contain a recognizable metric key (mAP/precision/recall — exact
  schema is the model's judgment call, not enforced), and a hard fail if no
  `results.csv`/`.pt` file exists anywhere under `runs/` — the strongest
  signal that `execute()` actually launched a real `yolo` process rather
  than the LLM fabricating the JSON files directly.
- **`CHANGES.md`** — this file.

## Other edits

- **`.env`** — added `SANDBOX_BACKEND=local` (with an explanation of the
  `local`/`docker` choice and why `MODAL_TOKEN_ID`/`SECRET` are now unused).
  Changed `LANGSMITH_TRACING` from `true` to `false`: it was `true` with a
  blank `LANGSMITH_API_KEY`, which meant every single LLM call was retrying
  a doomed trace upload to LangSmith and printing a 401 warning (confirmed
  via `ollama_test.py`'s output) — pure wasted latency with tracing
  effectively off anyway.
- **`README.md`** — added an "Running against Ollama" section, a
  "Sandbox (execute()) — do you need Docker?" section explaining the
  `local`/`docker` tradeoff, instructions for running both test scripts, and
  rewrote the "Current state / TODOs" section to reflect what's actually
  still stubbed (`zero_shot_annotate`, Roboflow/Kaggle creds, `dataset-agent`/
  `eval-agent` being untested) versus what's now fixed and covered.

## What's still a stub (unchanged, out of scope for this pass)

- `tools/zero_shot_annotate.py` — `NotImplementedError`, no detector wired in.
- Roboflow/Kaggle/Tavily credentials in `.env` are blank — `sourcing-agent`/
  `annotation-agent` degrade to an empty tool list without them (by design,
  see `tools/mcp_clients.py`'s own docstring).
- `dataset-agent` and `eval-agent` are wired the same way as
  `training-agent` now (dead `backend` key removed) but have no test script
  yet — untested end-to-end.

## Verification status

- `agent.py` — confirmed it imports cleanly and `create_deep_agent(...)`
  builds a compiled graph (`python -c "import agent"` succeeds post-fix).
- `test_planning_agent.py` — the `/skills` path fix was applied but the
  script was not re-run to completion (stopped mid-run at the user's
  request).
- `test_training_agent.py` — written but never executed.
- Docker path — Docker Desktop is installed on this machine but the daemon
  was not running when checked (`docker info` failed to connect), so
  `SANDBOX_BACKEND=docker` is untested. `local` is the one to try first.
