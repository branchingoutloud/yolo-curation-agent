"""YOLO eval driver - runs INSIDE the ultralytics GPU container only.

Invoked by tools/gpu_exec.py as:
    python /drivers/eval_driver.py <imgsz> <device> <split>

Reads /workspace/runs/train/weights/best.pt + /workspace/dataset/data.yaml,
runs validation on the GPU, and writes:
- /workspace/eval_report.md    (human-readable: per-class table + weak classes)
- /workspace/weak_classes.json (structured: class, metric, counts, likely cause)

Imports torch/ultralytics, so it runs only inside the container - never imported
by the orchestrator process.
"""

import json
import sys
import traceback
from pathlib import Path

_DATASET = Path("/workspace/dataset")
_BEST = Path("/workspace/runs/train/weights/best.pt")

# A class scoring below this mAP50 is flagged for diagnosis.
_WEAK_MAP50 = 0.5


def _names_list(cfg: dict) -> list[str]:
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return [str(n) for n in names]


def _count_instances(split: str, nc: int) -> list[int]:
    """Count label boxes per class id in dataset/labels/<split>/*.txt."""
    counts = [0] * nc
    labels_dir = _DATASET / "labels" / split
    if not labels_dir.is_dir():
        return counts
    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if 0 <= cid < nc:
                counts[cid] += 1
    return counts


def main() -> None:
    import yaml

    imgsz = int(sys.argv[1])
    device = sys.argv[2]
    split = sys.argv[3] if len(sys.argv) > 3 else "val"

    cfg = yaml.safe_load((_DATASET / "data.yaml").read_text(encoding="utf-8"))
    cfg["path"] = str(_DATASET)  # absolute - same fix as train_driver
    resolved = _DATASET / "data.resolved.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    names = _names_list(cfg)
    nc = len(names)

    from ultralytics import YOLO

    model = YOLO(str(_BEST))
    metrics = model.val(data=str(resolved), imgsz=imgsz, device=device, split=split)
    box = metrics.box

    ap50: dict[int, float] = {}
    ap: dict[int, float] = {}
    try:
        for i, class_index in enumerate(list(box.ap_class_index)):
            ap50[int(class_index)] = float(box.ap50[i])
            ap[int(class_index)] = float(box.ap[i])
    except Exception:  # noqa: BLE001 - best-effort per-class extraction
        pass

    train_counts = _count_instances("train", nc)
    val_counts = _count_instances(split, nc)
    positive = [c for c in train_counts if c > 0]
    median_train = sorted(positive)[len(positive) // 2] if positive else 0
    thin_floor = max(1, int(0.3 * median_train))

    weak_classes = []
    for cid in range(nc):
        cls_ap50 = ap50.get(cid, 0.0)
        n_train = train_counts[cid]
        is_thin = n_train < thin_floor
        if cls_ap50 >= _WEAK_MAP50 and not is_thin:
            continue
        causes = []
        if is_thin:
            causes.append(f"under-represented in training ({n_train} boxes vs ~{median_train} median)")
        if cls_ap50 < _WEAK_MAP50:
            causes.append("low mAP50 - check label quality / add harder or more varied examples")
        weak_classes.append(
            {
                "class": names[cid],
                "map50": round(cls_ap50, 4),
                "map50_95": round(ap.get(cid, 0.0), 4),
                "train_instances": n_train,
                "val_instances": val_counts[cid],
                "likely_cause": "; ".join(causes) or "below threshold",
            }
        )

    weak_json = {
        "overall_map50": float(box.map50),
        "overall_map50_95": float(box.map),
        "split": split,
        "weak_classes": weak_classes,
    }
    Path("/workspace/weak_classes.json").write_text(json.dumps(weak_json, indent=2), encoding="utf-8")

    confusion = None
    try:
        confusion = metrics.confusion_matrix.matrix.tolist()
    except Exception:  # noqa: BLE001 - confusion matrix is a nice-to-have
        confusion = None

    lines = [
        "# Eval report",
        "",
        f"- Split evaluated: `{split}`",
        f"- Overall mAP50: **{float(box.map50):.4f}**",
        f"- Overall mAP50-95: **{float(box.map):.4f}**",
        f"- Precision: {float(box.mp):.4f}  |  Recall: {float(box.mr):.4f}",
        "",
        "## Per-class",
        "",
        "| class | mAP50 | mAP50-95 | train boxes | val boxes |",
        "|---|---|---|---|---|",
    ]
    for cid in range(nc):
        lines.append(
            f"| {names[cid]} | {ap50.get(cid, 0.0):.3f} | {ap.get(cid, 0.0):.3f} "
            f"| {train_counts[cid]} | {val_counts[cid]} |"
        )
    lines += ["", "## Weak classes", ""]
    if weak_classes:
        for w in weak_classes:
            lines.append(f"- **{w['class']}** (mAP50 {w['map50']}): {w['likely_cause']}")
    else:
        lines.append("None flagged - every class is above the mAP50 threshold and adequately represented.")
    if confusion is not None:
        lines += [
            "",
            "## Confusion matrix",
            "",
            "Raw ultralytics matrix (last row/col = background). Large off-diagonal "
            "entries between two classes indicate they are being confused for each other.",
            "",
            "```json",
            json.dumps(confusion),
            "```",
        ]
    Path("/workspace/eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"EVAL_DONE map50={float(box.map50):.4f} weak_classes={len(weak_classes)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - surface full traceback to the caller's log
        traceback.print_exc()
        sys.exit(1)
