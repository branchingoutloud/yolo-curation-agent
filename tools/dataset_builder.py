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
is required per YOLO-format source because YOLO label files reference classes
by numeric ID only - IDs are not portable across independently-annotated
sources, so merging without remapping through each source's own class list
would silently scramble labels.

Pascal VOC XML is also supported as an alternative to YOLO .txt labels - put
`.xml` files (same stem as the image) in `labels/` instead, no `classes.txt`
needed (VOC names each object's class directly as a string). Added after a
live run hit a real academic dataset (Stanford Dogs) shipping in this format,
not YOLO - see `_parse_pascal_voc_xml` for the conversion and its single-
canonical-class special case.
"""

from __future__ import annotations

import json
import mimetypes
import random
import shutil
import tarfile
import xml.etree.ElementTree as ET
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


def _parse_pascal_voc_xml(
    xml_path: Path,
    canonical_classes: list[str],
    canonical_index: dict[str, int],
) -> list[str] | None:
    """Convert one Pascal VOC XML annotation into YOLO-format label lines.

    Pascal VOC (the format Stanford Dogs and many other academic datasets ship
    in) names each <object>'s class directly as a string and gives absolute
    pixel bounding boxes - unlike YOLO, there's no numeric class-ID/classes.txt
    indirection to resolve first.

    Single-canonical-class special case: Stanford Dogs' own <name> fields are
    the specific BREED (e.g. "chihuahua"), not "dog" - it's a fine-grained
    120-breed dataset, not a generic dog/no-dog one. For single-class
    detection (the common case here - one canonical class covering "any
    breed"), matching <name> against the canonical class list would silently
    drop every box. So: if there's exactly one canonical class, every object
    in the file is treated as that class regardless of its literal <name> -
    this is a real single-class detection use case, not a bug. With multiple
    canonical classes, falls back to a case-insensitive exact match between
    <name> and a canonical class name.

    Returns the converted lines, or None if no object produced a valid box
    (image should be dropped, same convention as `_remap_label_file`).
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    size = root.find("size")
    if size is None:
        return None
    try:
        img_w = float(size.findtext("width", ""))
        img_h = float(size.findtext("height", ""))
    except ValueError:
        return None
    if img_w <= 0 or img_h <= 0:
        return None

    single_class_id = 0 if len(canonical_classes) == 1 else None

    lines: list[str] = []
    for obj in root.findall("object"):
        if single_class_id is not None:
            canonical_id = single_class_id
        else:
            name = (obj.findtext("name") or "").strip().lower()
            canonical_id = canonical_index.get(name)
            if canonical_id is None:
                continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext("xmin", ""))
            ymin = float(bndbox.findtext("ymin", ""))
            xmax = float(bndbox.findtext("xmax", ""))
            ymax = float(bndbox.findtext("ymax", ""))
        except ValueError:
            continue

        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        if width <= 0 or height <= 0:
            continue
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h

        lines.append(f"{canonical_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines or None


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
    max_total_images: int | None = None,
) -> str:
    """Merge locally-staged sources into one deduped, split YOLO dataset.

    Reads `sources_json_path` and, for each entry with `status == "available"`,
    looks for that entry's raw files under `sourced_dir/<index>/` (images/,
    labels/ - see this module's docstring for the exact convention). Entries
    with any other status (e.g. `needs_annotation`) are skipped and reported,
    not silently dropped - annotation-agent isn't part of this run, so those
    stay pending.

    The annotation format actually present in `labels/` is auto-detected per
    source (NOT trusted from sources.json's free-text `annotation_format`
    field, which has been inconsistent in practice) - `.txt` files are treated
    as YOLO (numeric class IDs, requires a `classes.txt` alongside to remap
    them safely) and `.xml` files as Pascal VOC (class named directly per
    object, no classes.txt needed - and if there's exactly one canonical
    class, every object is treated as that class regardless of its literal
    name, since fine-grained academic datasets like Stanford Dogs label the
    specific breed, not a generic "dog" - see `_parse_pascal_voc_xml`). A
    source with neither is skipped and reported, not silently dropped.

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

        source_dir = sourced_root / str(i)
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            skipped.append(f"{label}: no local files at {sourced_dir}/{i}/ yet - fetch it first (Roboflow/Kaggle MCP tools or download_and_extract)")
            continue

        # Auto-detect annotation format from what's actually staged, rather
        # than trusting sources.json's free-text annotation_format string
        # (observed inconsistent values in practice: "YOLO (Roboflow)",
        # "bounding boxes, class labels", "unknown", etc.) - YOLO .txt files
        # need classes.txt for numeric-ID remapping; Pascal VOC .xml files
        # name classes directly as strings and don't.
        has_yolo_labels = any(labels_dir.glob("*.txt"))
        has_voc_labels = any(labels_dir.glob("*.xml"))
        local_classes = _read_classes_txt(source_dir) if has_yolo_labels else None
        if has_yolo_labels and local_classes is None:
            skipped.append(f"{label}: has .txt label files but no classes.txt at {sourced_dir}/{i}/ - can't remap numeric class IDs safely")
            continue
        if not has_yolo_labels and not has_voc_labels:
            skipped.append(f"{label}: no recognized label files (.txt or .xml) at {sourced_dir}/{i}/labels/ yet")
            continue

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if has_yolo_labels:
                label_path = labels_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue
                remapped = _remap_label_file(label_path, local_classes, canonical_index)
            else:
                label_path = labels_dir / f"{image_path.stem}.xml"
                if not label_path.exists():
                    continue
                remapped = _parse_pascal_voc_xml(label_path, canonical_classes, canonical_index)
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
    total_before_cap = len(kept)
    if max_total_images is not None and max_total_images > 0:
        kept = kept[:max_total_images]
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
        # Absolute, not "." - ultralytics resolves a relative `path` against
        # whatever cwd `yolo detect train` happens to be invoked from, not
        # against data.yaml's own directory. Confirmed live: training-agent's
        # execute() runs with cwd=RUN_ARTIFACTS_DIR (this dataset's PARENT,
        # not this directory itself), so "." resolved to the wrong directory
        # and ultralytics reported images "not found" one level up from
        # where they actually are. An absolute path is correct regardless of
        # invocation cwd.
        "path": str(out_root),
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
    if max_total_images is not None and total_before_cap > len(kept):
        summary_lines.append(
            f"Capped at max_total_images={max_total_images} ({total_before_cap} unique images were "
            f"available after dedup; a random subset - seeded, reproducible - was kept)."
        )
    if skipped:
        summary_lines.append("Skipped sources:")
        summary_lines.extend(f"- {s}" for s in skipped)
    return "\n".join(summary_lines)
