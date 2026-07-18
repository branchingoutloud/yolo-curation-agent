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
`download_and_extract` below, before calling `merge_and_split_dataset`) - two
layouts are recognized:

  flat (e.g. a manually-staged or Kaggle source):
    /workspace/sourced/<i>/images/*.{jpg,jpeg,png}
    /workspace/sourced/<i>/labels/*.txt      (YOLO format, same stem as image)
    /workspace/sourced/<i>/classes.txt       (one class name per line, in the
                                               numeric-ID order that source's
                                               label files use)

  Roboflow YOLOv8 export (what versions_export(export_format="yolov8") +
  download_and_extract actually produces - real exports never contain a bare
  classes.txt, only data.yaml, and always pre-split into train/valid/test):
    /workspace/sourced/<i>/data.yaml         (`names:` list or {id: name} dict)
    /workspace/sourced/<i>/{train,valid,test}/images/*.{jpg,jpeg,png}
    /workspace/sourced/<i>/{train,valid,test}/labels/*.txt

`<i>` is the 0-based index of that entry in sources.json's array. A source's
own class list (however it's spelled) is required because YOLO label files
reference classes by numeric ID only - IDs are not portable across
independently-annotated sources, so merging without remapping through each
source's own class list would silently scramble labels. A Roboflow source's
own train/valid/test split is intentionally NOT preserved - every image is
pooled and this module re-splits across the whole merged/deduped set itself.
"""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger("dataset_agent")

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
    logger.info("download_and_extract: url=%s dest=%s", url, dest_dir)
    try:
        dest = _workspace_path(dest_dir)
    except ValueError as exc:
        logger.error("download_and_extract: invalid dest path %s: %s", dest_dir, exc)
        return f"Error: {exc}. Pass a path starting with /workspace/ instead."
    dest.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        logger.info("download_and_extract: HTTP %s content-type=%s", response.status_code, response.headers.get('Content-Type', '?'))
    except requests.RequestException as exc:
        logger.error("download_and_extract: request failed for %s: %s", url, exc)
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
                logger.error("download_and_extract: exceeded %dMB guardrail for %s", _MAX_DOWNLOAD_BYTES // (1024 * 1024), url)
                return f"Download aborted: {url} exceeded the {_MAX_DOWNLOAD_BYTES // (1024 * 1024)}MB guardrail."
            f.write(chunk)

    if zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path) as zf:
            zf.extractall(dest)
        download_path.unlink()
        logger.info("download_and_extract: extracted zip, %d bytes from %s", total, url)
        return f"Downloaded and extracted {total} bytes from {url} into {dest_dir} (zip archive)."

    try:
        if tarfile.is_tarfile(download_path):
            with tarfile.open(download_path) as tf:
                tf.extractall(dest)  # noqa: S202 - dest is confined to the workspace root by _workspace_path
            download_path.unlink()
            logger.info("download_and_extract: extracted tar, %d bytes from %s", total, url)
            return f"Downloaded and extracted {total} bytes from {url} into {dest_dir} (tar archive)."
    except tarfile.TarError:
        pass

    logger.info("download_and_extract: saved raw file %s/%s (%d bytes)", dest_dir, download_name, total)
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
def list_qualifying_sources(
    sources_json_path: str = "/workspace/sources.json",
    max_sources: int = 3,
) -> str:
    """List which sources to fetch/merge right now, capped at max_sources.

    Filters sources_json_path down to entries with status == "available" and
    annotation_format == "YOLO" (the same filter merge_and_split_dataset
    applies) - real filtering over the file, not something the calling agent
    has to eyeball itself across a long sources.json. Sorts by image_count
    descending and returns only the top `max_sources` (default 3) - forking
    and exporting every qualifying source has real Roboflow API cost (a fork
    + version-generate + export per source) for shrinking benefit once the
    largest few are already merged, so this is capped rather than unlimited.
    Enforced here in code rather than left to the calling agent to
    self-limit, since a prompted "only fetch N of these" instruction is not
    reliably followed.

    Returns a plain-text list of the selected sources (0-based index,
    source, dataset_id, url, image_count), largest-first, plus a separate
    list of any additional qualifying sources excluded purely by this cap
    (named, not silently dropped, so the calling agent can mention them in
    its summary) - or a plain-text explanation if none currently qualify.
    Fetch/stage EVERY listed (selected) index under sourced_dir/<index>/ -
    merge_and_split_dataset merges whatever is staged across all of them
    (deduping near-identical images and re-splitting the combined pool), and
    skips anything left unstaged rather than erroring, so skipping a
    qualifying source just means a smaller merged dataset, not a failure.
    """
    logger.info("list_qualifying_sources: reading %s (max_sources=%d)", sources_json_path, max_sources)
    try:
        sources = _load_sources(sources_json_path)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("list_qualifying_sources: failed to load sources: %s", exc)
        return f"Error: {exc}"

    candidates = [
        (i, source)
        for i, source in enumerate(sources)
        if source.get("status") == "available"
        and str(source.get("annotation_format", "")).strip().lower() == "yolo"
    ]
    logger.info("list_qualifying_sources: %d total sources, %d qualify (available + YOLO)", len(sources), len(candidates))
    for i, src in enumerate(sources):
        logger.debug(
            "  source[%d]: id=%s status=%s format=%s image_count=%s",
            i, src.get('dataset_id'), src.get('status'), src.get('annotation_format'), src.get('image_count'),
        )
    if not candidates:
        logger.warning("list_qualifying_sources: no qualifying candidates found")
        return (
            "No source currently qualifies (need status == 'available' and "
            "annotation_format == 'YOLO'). Nothing to fetch - report this back "
            "rather than fetching anything."
        )

    candidates.sort(key=lambda pair: pair[1].get("image_count", 0) or 0, reverse=True)
    selected = candidates[:max_sources]
    dropped = candidates[max_sources:]
    logger.info(
        "list_qualifying_sources: %d qualifying, selecting top %d (indices=%s)%s",
        len(candidates), len(selected), [i for i, _ in selected],
        f" - dropped by cap: indices={[i for i, _ in dropped]}" if dropped else "",
    )
    lines = [
        f"{len(selected)} of {len(candidates)} qualifying source(s) selected (largest first, "
        f"capped at max_sources={max_sources}) - fetch/stage EVERY index below, then call "
        "merge_and_split_dataset once:"
    ]
    for i, source in selected:
        lines.append(
            f"  index {i}: source={source.get('source')!r} dataset_id={source.get('dataset_id')!r} "
            f"url={source.get('url')!r} image_count={source.get('image_count')!r} "
            f"classes_covered={source.get('classes_covered')!r}"
        )
    if dropped:
        lines.append(
            f"{len(dropped)} additional qualifying source(s) NOT selected due to the "
            f"max_sources={max_sources} cap - do not fetch these, but mention them in your "
            "summary so the orchestrator knows more data exists for a future round:"
        )
        for i, source in dropped:
            lines.append(
                f"  index {i}: source={source.get('source')!r} dataset_id={source.get('dataset_id')!r} "
                f"image_count={source.get('image_count')!r}"
            )
    return "\n".join(lines)


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


_ROBOFLOW_SPLIT_DIRS = ("train", "valid", "val", "test")


def _read_classes(source_dir: Path) -> list[str] | None:
    """A source's class list, in numeric-ID order - tolerant of a flat
    classes.txt (this module's own manually-staged convention) or a Roboflow
    YOLOv8 export's data.yaml (`names:` as a list or a {id: name} dict) - a
    real Roboflow export never contains a bare classes.txt, only data.yaml.
    """
    classes_file = source_dir / "classes.txt"
    if classes_file.exists():
        with open(classes_file, encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        logger.debug("_read_classes: %s -> %d classes from classes.txt: %s", source_dir.name, len(classes), classes)
        return classes

    data_yaml = source_dir / "data.yaml"
    if data_yaml.exists():
        try:
            with open(data_yaml, encoding="utf-8") as f:
                parsed = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.warning("_read_classes: %s is not valid YAML: %s", data_yaml, exc)
            return None
        names = parsed.get("names") if isinstance(parsed, dict) else None
        if isinstance(names, list):
            logger.debug("_read_classes: %s -> %d classes from data.yaml (list): %s", source_dir.name, len(names), names)
            return list(names)
        if isinstance(names, dict):
            classes = [names[k] for k in sorted(names, key=int)]
            logger.debug("_read_classes: %s -> %d classes from data.yaml (dict): %s", source_dir.name, len(classes), classes)
            return classes
        logger.warning("_read_classes: %s has no usable 'names' field - source cannot be merged", data_yaml)
        return None

    logger.warning("_read_classes: no classes.txt or data.yaml under %s - source cannot be merged", source_dir)
    return None


def _iter_image_label_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """(image_path, label_path) pairs for a staged source - tolerant of a flat
    images/+labels/ layout or a Roboflow-style pre-split train/valid/test
    (each with its own images/+labels/). A source's own pre-applied split is
    not preserved here; merge_and_split_dataset re-splits the merged pool
    itself, so every split's images are pooled together at this stage.
    """
    flat_images, flat_labels = source_dir / "images", source_dir / "labels"
    if flat_images.is_dir():
        search_roots = [(flat_images, flat_labels)]
    else:
        search_roots = [
            (source_dir / split / "images", source_dir / split / "labels")
            for split in _ROBOFLOW_SPLIT_DIRS
            if (source_dir / split / "images").is_dir()
        ]

    pairs: list[tuple[Path, Path]] = []
    for images_dir, labels_dir in search_roots:
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                pairs.append((image_path, labels_dir / f"{image_path.stem}.txt"))
    return pairs


def _remap_label_file(
    label_path: Path,
    local_classes: list[str],
    canonical_index: dict[str, int],
) -> list[str] | None:
    """Rewrite a YOLO label file's class IDs into the canonical class list.

    `canonical_index` must be keyed by `.strip().casefold()`d class names -
    different dataset authors spell the same class differently ("Kangaroo" vs
    "kangaroo" vs "KANGAROO"; observed live across sourcing-agent's own
    entries for the same search), and an exact-string lookup here would
    silently drop every box in every label file as "unmapped" the moment
    casing differs, with no error - just an empty merged dataset.

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
            canonical_id = canonical_index.get(class_name.strip().casefold())
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
    logger.info("merge_and_split_dataset: starting (sources=%s, sourced=%s, output=%s)", sources_json_path, sourced_dir, output_dir)
    try:
        import imagehash
        from PIL import Image
    except ImportError as exc:
        logger.error("merge_and_split_dataset: missing dependency: %s", exc)
        return f"Missing dependency ({exc}); run `uv sync` after adding pillow/imagehash to pyproject.toml."

    try:
        sources = _load_sources(sources_json_path)
        canonical_classes = _canonical_class_list(class_budget_path, sources)
        canonical_index = {name.strip().casefold(): i for i, name in enumerate(canonical_classes)}
        sourced_root = _workspace_path(sourced_dir)
        out_root = _workspace_path(output_dir)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("merge_and_split_dataset: setup failed: %s", exc)
        return f"Error: {exc}"

    logger.info("merge_and_split_dataset: canonical_classes=%s (%d total)", canonical_classes, len(canonical_classes))

    skipped: list[str] = []
    kept_hashes: list["imagehash.ImageHash"] = []
    duplicates_dropped = 0
    kept: list[tuple[Path, list[str]]] = []  # (image_path, remapped_label_lines)

    for i, source in enumerate(sources):
        label = f"{source.get('source', '?')}/{source.get('dataset_id', '?')}"
        status = source.get("status")
        logger.info("merge_and_split_dataset: source[%d] %s status=%s format=%s", i, label, status, source.get('annotation_format'))
        if status != "available":
            reason = f"{label}: status={status!r} - not merged (needs annotation-agent, which isn't active this run)"
            skipped.append(reason)
            logger.warning("merge_and_split_dataset: SKIPPED source[%d]: %s", i, reason)
            continue
        annotation_format = str(source.get("annotation_format", "")).strip().lower()
        if annotation_format != "yolo":
            reason = f"{label}: annotation_format={source.get('annotation_format')!r} not supported yet (only YOLO-format sources auto-merge)"
            skipped.append(reason)
            logger.warning("merge_and_split_dataset: SKIPPED source[%d]: %s", i, reason)
            continue

        source_dir = sourced_root / str(i)
        local_classes = _read_classes(source_dir)
        image_label_pairs = _iter_image_label_pairs(source_dir) if local_classes is not None else []
        if local_classes is None or not image_label_pairs:
            reason = f"{label}: no local files at {sourced_dir}/{i}/ yet - fetch it first (Roboflow/Kaggle MCP tools or download_and_extract)"
            skipped.append(reason)
            logger.warning("merge_and_split_dataset: SKIPPED source[%d]: %s", i, reason)
            continue

        src_img_count, src_label_match, src_remap_ok, src_dedup = 0, 0, 0, 0
        for image_path, label_path in image_label_pairs:
            src_img_count += 1
            if not label_path.exists():
                logger.debug("merge: source[%d] no label for %s", i, image_path.name)
                continue
            src_label_match += 1
            remapped = _remap_label_file(label_path, local_classes, canonical_index)
            if remapped is None:
                logger.debug("merge: source[%d] all classes unmapped in %s", i, label_path.name)
                continue
            src_remap_ok += 1

            try:
                with Image.open(image_path) as img:
                    img_hash = imagehash.phash(img)
            except OSError as exc:
                logger.warning("merge: source[%d] cannot open %s: %s", i, image_path.name, exc)
                continue

            if any((img_hash - kept_hash) <= dedup_hash_threshold for kept_hash in kept_hashes):
                duplicates_dropped += 1
                src_dedup += 1
                logger.debug("merge: source[%d] near-duplicate dropped: %s", i, image_path.name)
                continue

            kept_hashes.append(img_hash)
            kept.append((image_path, remapped))

        logger.info(
            "merge_and_split_dataset: source[%d] result: images=%d labels_matched=%d remapped=%d dedup_dropped=%d kept=%d",
            i, src_img_count, src_label_match, src_remap_ok, src_dedup, src_remap_ok - src_dedup,
        )

    if not kept:
        logger.error("merge_and_split_dataset: no images merged - nothing to build a dataset from")
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
    logger.info(
        "merge_and_split_dataset: split %d images -> train=%d val=%d test=%d (%.0f/%.0f/%.0f%%), %d duplicates dropped",
        len(kept), len(splits['train']), len(splits['val']), len(splits['test']),
        train_ratio * 100, val_ratio * 100, _test_ratio * 100, duplicates_dropped,
    )

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
    logger.info("merge_and_split_dataset: wrote data.yaml to %s (nc=%d, classes=%s)", output_dir, len(canonical_classes), canonical_classes)
    logger.info("merge_and_split_dataset: per-class box counts: %s", class_counts)

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
