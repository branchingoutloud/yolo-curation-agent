"""Custom tools for dataset-agent: real download + merge/dedupe/split logic.

Why these are plain local tools and not execute()-in-a-sandbox calls: the
installed `deepagents` version (0.6.12) does not support a per-subagent
`"backend"` override - `create_deep_agent`'s `SubAgentMiddleware` always binds
every subagent's `FilesystemMiddleware` to the single `backend=` passed to
`create_deep_agent` itself (see `graph.py`'s subagent-building loop), so the
`"backend": sandbox_backend` field on `annotation-agent`/`training-agent`/
`eval-agent`'s `SubAgent` dicts is inert - it's not a field `SubAgent`
declares, and nothing reads it per-subagent. There is currently no supported
way to give one subagent its own sandbox. That's fine here: merging,
deduping, and splitting a dataset is pure CPU-bound file/image work (no GPU,
no ultralytics), so it doesn't need a sandbox at all - these tools just run
directly in the orchestrator process, the same pattern as `zero_shot_annotate`.

Directory convention this module assumes (dataset-agent is responsible for
getting sources into this shape, e.g. via Roboflow/Kaggle MCP tools or
`download_and_extract` below, before calling `merge_and_split_dataset`):

    /workspace/sourced/<i>/images/*.{jpg,jpeg,png}
    /workspace/sourced/<i>/labels/*.txt      (YOLO format, same stem as image)
    /workspace/sourced/<i>/classes.txt       (one class name per line, in the
                                               numeric-ID order that source's
                                               label files use)

`<i>` is the 0-based index of that entry in sources.json's array. `classes.txt`
is required per source because YOLO label files reference classes by numeric
ID only - IDs are not portable across independently-annotated sources, so
merging without remapping through each source's own class list would silently
scramble labels.
"""

from __future__ import annotations

import json
import mimetypes
import random
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
import yaml
from langchain_core.tools import tool

from tools.workspace_paths import workspace_path as _workspace_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2GB guardrail against runaway downloads


@tool
def download_and_extract(url: str, dest_dir: str) -> str:
    """Download a file from a direct URL and extract it if it's an archive.

    Use this for a Kaggle direct-download link, a Roboflow export URL, or any
    other plain HTTP(S) dataset URL. `dest_dir` must be a `/workspace/...`
    path (e.g. `/workspace/sourced/1/`); it's created if missing. Zip and
    tar(.gz) archives are extracted in place; anything else is saved as-is
    under `dest_dir`. Returns a short summary of what was written.
    """
    try:
        dest = _workspace_path(dest_dir)
    except ValueError as exc:
        return f"Error: {exc}. Pass a path starting with /workspace/ instead."
    dest.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"Download failed for {url}: {exc}"

    content_type = response.headers.get("Content-Type", "")
    guessed_ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
    download_name = Path(url.split("?")[0]).name or f"download{guessed_ext or '.bin'}"
    download_path = dest / download_name

    total = 0
    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            total += len(chunk)
            if total > _MAX_DOWNLOAD_BYTES:
                download_path.unlink(missing_ok=True)
                return f"Download aborted: {url} exceeded the {_MAX_DOWNLOAD_BYTES // (1024 * 1024)}MB guardrail."
            f.write(chunk)

    if zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path) as zf:
            zf.extractall(dest)
        download_path.unlink()
        return f"Downloaded and extracted {total} bytes from {url} into {dest_dir} (zip archive)."

    try:
        if tarfile.is_tarfile(download_path):
            with tarfile.open(download_path) as tf:
                tf.extractall(dest)  # noqa: S202 - dest is confined to the workspace root by _workspace_path
            download_path.unlink()
            return f"Downloaded and extracted {total} bytes from {url} into {dest_dir} (tar archive)."
    except tarfile.TarError:
        pass

    return f"Downloaded {total} bytes from {url} to {dest_dir}/{download_name} (not an archive, saved as-is)."


def _load_sources(sources_json_path: str) -> list[dict]:
    path = _workspace_path(sources_json_path)
    if not path.exists():
        msg = f"{sources_json_path} does not exist - sourcing-agent must run first."
        raise FileNotFoundError(msg)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        msg = f"{sources_json_path} must contain a JSON array of source entries."
        raise ValueError(msg)
    return data


@tool
def select_primary_source(sources_json_path: str = "/workspace/sources.json") -> str:
    """Pick the single mergeable source with the highest image_count.

    Filters sources_json_path down to entries with status == "available" and
    annotation_format == "YOLO" (the same filter merge_and_split_dataset
    applies), and returns the one with the largest image_count as the sole
    dataset to fetch/stage/train on - real arithmetic over the file, not a
    count the calling agent has to eyeball itself across a long sources.json.

    Returns a plain-text description of the chosen source (its 0-based index,
    source, dataset_id, url, image_count) or a plain-text explanation if no
    entry currently qualifies. Only fetch/stage the returned index's files
    under sourced_dir/<index>/ - leave every other source unfetched, so
    merge_and_split_dataset naturally builds the dataset from this one source
    alone (it skips anything not staged rather than erroring).
    """
    try:
        sources = _load_sources(sources_json_path)
    except (ValueError, FileNotFoundError) as exc:
        return f"Error: {exc}"

    candidates = [
        (i, source)
        for i, source in enumerate(sources)
        if source.get("status") == "available"
        and str(source.get("annotation_format", "")).strip().lower() == "yolo"
    ]
    if not candidates:
        return (
            "No source currently qualifies (need status == 'available' and "
            "annotation_format == 'YOLO'). Nothing to select - report this back "
            "rather than fetching anything."
        )

    best_index, best_source = max(candidates, key=lambda pair: pair[1].get("image_count", 0) or 0)
    return (
        f"Selected index {best_index}: source={best_source.get('source')!r} "
        f"dataset_id={best_source.get('dataset_id')!r} url={best_source.get('url')!r} "
        f"image_count={best_source.get('image_count')!r} classes_covered={best_source.get('classes_covered')!r}. "
        f"Fetch/stage only this index's files under sourced_dir/{best_index}/, then call "
        f"merge_and_split_dataset - do not stage any other source."
    )


def _canonical_class_list(class_budget_path: str, sources: list[dict]) -> list[str]:
    """Best-effort class list, tolerant of planning-agent's exact JSON shape.

    Falls back to the union of every source's `classes_covered` (sorted for
    determinism) if class_budget.json is missing or in an unrecognized shape -
    dataset-agent shouldn't hard-fail just because it can't parse a sibling
    subagent's file.
    """
    path = _workspace_path(class_budget_path)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                budget = json.load(f)
            if isinstance(budget, list) and all(isinstance(c, str) for c in budget):
                return budget
            if isinstance(budget, dict):
                if isinstance(budget.get("classes"), list):
                    return list(budget["classes"])
                if all(isinstance(v, (int, float, dict)) for v in budget.values()):
                    return list(budget.keys())
        except (json.JSONDecodeError, OSError):
            pass

    classes: set[str] = set()
    for source in sources:
        classes.update(source.get("classes_covered", []))
    return sorted(classes)


def _read_classes_txt(source_dir: Path) -> list[str] | None:
    classes_file = source_dir / "classes.txt"
    if not classes_file.exists():
        return None
    with open(classes_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _remap_label_file(
    label_path: Path,
    local_classes: list[str],
    canonical_index: dict[str, int],
) -> list[str] | None:
    """Rewrite a YOLO label file's class IDs into the canonical class list.

    Returns the remapped lines, or `None` if every box referenced a class not
    in the canonical list (image should be dropped rather than kept with zero
    valid boxes).
    """
    remapped: list[str] = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                local_id = int(parts[0])
            except ValueError:
                continue
            if local_id < 0 or local_id >= len(local_classes):
                continue
            class_name = local_classes[local_id]
            canonical_id = canonical_index.get(class_name)
            if canonical_id is None:
                continue
            remapped.append(" ".join([str(canonical_id), *parts[1:]]))
    return remapped or None


def _split_ratios(total: int) -> tuple[float, float, float]:
    """Train/val/test ratios per the cv-dataset-curation skill's split guidance."""
    if total >= 5000:
        return (0.8, 0.1, 0.1)
    return (0.7, 0.2, 0.1)


@tool
def merge_and_split_dataset(
    sources_json_path: str = "/workspace/sources.json",
    sourced_dir: str = "/workspace/sourced",
    output_dir: str = "/workspace/dataset",
    class_budget_path: str = "/workspace/class_budget.json",
    dedup_hash_threshold: int = 5,
    seed: int = 42,
) -> str:
    """Merge locally-staged sources into one deduped, split YOLO dataset.

    Reads `sources_json_path` and, for each entry with `status == "available"`
    and `annotation_format == "YOLO"`, looks for that entry's raw files under
    `sourced_dir/<index>/` (images/, labels/, classes.txt - see this module's
    docstring for the exact convention). Entries with any other status (e.g.
    `needs_annotation`) are skipped and reported, not silently dropped -
    annotation-agent isn't part of this run, so those stay pending.

    Remaps each source's YOLO class IDs into one canonical class list (from
    `class_budget_path`, or the union of every source's `classes_covered` if
    that file is missing/unrecognized), drops near-duplicate images via
    perceptual hashing, splits by whole image (never by crop, so augmented
    near-duplicates can't leak across splits) into train/val/test per the
    cv-dataset-curation skill's ratio guidance, and writes the YOLO-format
    directory tree plus `data.yaml` under `output_dir`.

    Returns a plain-text summary (final counts per split, sources skipped and
    why, images dropped as duplicates) - not JSON - so the calling agent can
    relay it close to verbatim as its final message.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError as exc:
        return f"Missing dependency ({exc}); run `uv sync` after adding pillow/imagehash to pyproject.toml."

    try:
        sources = _load_sources(sources_json_path)
        canonical_classes = _canonical_class_list(class_budget_path, sources)
        canonical_index = {name: i for i, name in enumerate(canonical_classes)}
        sourced_root = _workspace_path(sourced_dir)
        out_root = _workspace_path(output_dir)
    except (ValueError, FileNotFoundError) as exc:
        return f"Error: {exc}"

    skipped: list[str] = []
    kept_hashes: list["imagehash.ImageHash"] = []
    duplicates_dropped = 0
    kept: list[tuple[Path, list[str]]] = []  # (image_path, remapped_label_lines)

    for i, source in enumerate(sources):
        label = f"{source.get('source', '?')}/{source.get('dataset_id', '?')}"
        status = source.get("status")
        if status != "available":
            skipped.append(f"{label}: status={status!r} - not merged (needs annotation-agent, which isn't active this run)")
            continue
        annotation_format = str(source.get("annotation_format", "")).strip().lower()
        if annotation_format != "yolo":
            skipped.append(f"{label}: annotation_format={source.get('annotation_format')!r} not supported yet (only YOLO-format sources auto-merge)")
            continue

        source_dir = sourced_root / str(i)
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"
        local_classes = _read_classes_txt(source_dir)
        if not images_dir.is_dir() or not labels_dir.is_dir() or local_classes is None:
            skipped.append(f"{label}: no local files at {sourced_dir}/{i}/ yet - fetch it first (Roboflow/Kaggle MCP tools or download_and_extract)")
            continue

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            remapped = _remap_label_file(label_path, local_classes, canonical_index)
            if remapped is None:
                continue

            try:
                with Image.open(image_path) as img:
                    img_hash = imagehash.phash(img)
            except OSError:
                continue

            if any((img_hash - kept_hash) <= dedup_hash_threshold for kept_hash in kept_hashes):
                duplicates_dropped += 1
                continue

            kept_hashes.append(img_hash)
            kept.append((image_path, remapped))

    if not kept:
        summary_lines = [
            "No images merged - nothing available to build a dataset from.",
            *(f"- {s}" for s in skipped),
        ]
        return "\n".join(summary_lines)

    rng = random.Random(seed)
    rng.shuffle(kept)
    train_ratio, val_ratio, _test_ratio = _split_ratios(len(kept))
    n_train = round(len(kept) * train_ratio)
    n_val = round(len(kept) * val_ratio)
    splits = {
        "train": kept[:n_train],
        "val": kept[n_train : n_train + n_val],
        "test": kept[n_train + n_val :],
    }

    for split_name, items in splits.items():
        (out_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    class_counts: dict[str, int] = {name: 0 for name in canonical_classes}
    for split_name, items in splits.items():
        for idx, (image_path, remapped_lines) in enumerate(items):
            safe_name = f"{split_name}_{idx}{image_path.suffix.lower()}"
            shutil.copyfile(image_path, out_root / "images" / split_name / safe_name)
            label_out = out_root / "labels" / split_name / f"{Path(safe_name).stem}.txt"
            label_out.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
            for line in remapped_lines:
                class_id = int(line.split()[0])
                if 0 <= class_id < len(canonical_classes):
                    class_counts[canonical_classes[class_id]] += 1

    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(canonical_classes),
        "names": {i: name for i, name in enumerate(canonical_classes)},
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    summary_lines = [
        f"Merged {len(kept)} images ({duplicates_dropped} near-duplicates dropped) into {output_dir}.",
        f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])} "
        f"({train_ratio:.0%}/{val_ratio:.0%}/{1 - train_ratio - val_ratio:.0%}).",
        f"Per-class box counts (post-merge, all splits): {class_counts}",
    ]
    if skipped:
        summary_lines.append("Skipped sources:")
        summary_lines.extend(f"- {s}" for s in skipped)
    return "\n".join(summary_lines)
