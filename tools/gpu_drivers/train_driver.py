"""YOLO training driver - runs INSIDE the ultralytics GPU container only.

Invoked by tools/gpu_exec.py as:
    python /drivers/train_driver.py <model_size> <epochs> <imgsz> <batch> <device>

Reads /workspace/dataset/data.yaml (produced by dataset-agent), trains on the
GPU, and writes /workspace/runs/train/metrics.json plus weights/best.pt. This
file imports torch/ultralytics, so it must never be imported by the orchestrator
process - it is executed only inside the container where those exist.
"""

import json
import sys
import traceback
from pathlib import Path

_DATASET = Path("/workspace/dataset")
_RUNS = Path("/workspace/runs")


def _resolve_data_yaml() -> str:
    """Rewrite data.yaml's relative `path: "."` to an absolute path.

    dataset-agent writes `path: "."`, but ultralytics resolves a relative
    `path` against its own `datasets_dir` setting (NOT the yaml's location), so
    training would fail to find images. We emit a sibling data.resolved.yaml
    with an absolute `path` and train on that; the original is left untouched.
    """
    import yaml

    src = _DATASET / "data.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["path"] = str(_DATASET)
    resolved = _DATASET / "data.resolved.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(resolved)


def main() -> None:
    model_size, epochs, imgsz, batch, device = (
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        sys.argv[5],
    )
    data = _resolve_data_yaml()

    from ultralytics import YOLO, settings

    # Force ultralytics to write ALL run outputs under the (writable, bind-
    # mounted, user-owned) /workspace/runs. The image's default runs_dir is
    # /ultralytics/runs, which is root-owned - and we run as a non-root --user,
    # so any ultralytics call that falls back to the default (e.g. a .val()
    # without an explicit project=) would hit PermissionError.
    settings.update({"runs_dir": "/workspace/runs"})

    model = YOLO(f"{model_size}.pt")
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(_RUNS),
        name="train",
        exist_ok=True,
        device=device,
        workers=8,
        patience=20,
    )

    # Validate the BEST checkpoint (model still holds `last` after train()).
    best = _RUNS / "train" / "weights" / "best.pt"
    val_model = YOLO(str(best)) if best.exists() else model
    metrics = val_model.val(
        data=data,
        imgsz=imgsz,
        device=device,
        split="val",
        project=str(_RUNS),
        name="val",
        exist_ok=True,
    )
    box = metrics.box

    per_class: dict[str, dict[str, float]] = {}
    try:
        for i, class_index in enumerate(list(box.ap_class_index)):
            name = str(metrics.names[int(class_index)])
            per_class[name] = {
                "map50": float(box.ap50[i]),
                "map50_95": float(box.ap[i]),
            }
    except Exception:  # noqa: BLE001 - per-class breakdown is best-effort
        pass

    out = {
        "model_size": model_size,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
        "best_weights": str(best),
    }
    (_RUNS / "train").mkdir(parents=True, exist_ok=True)
    (_RUNS / "train" / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("METRICS_JSON " + json.dumps(out))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - surface full traceback to the caller's log
        traceback.print_exc()
        sys.exit(1)
