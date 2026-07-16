---
name: cv-eval-and-iteration
description: Read YOLO training metrics (mAP50/mAP50-95, precision/recall)
  and confusion-matrix patterns, and diagnose a likely cause per weak class
  (too little data, label noise, or class confusability). Use after a
  training run when writing eval_report.md and weak_classes.json.
license: MIT
---

# CV eval and iteration diagnosis

## Reading the headline metrics
- **mAP50** - detection quality at a loose IoU threshold (0.5). Good proxy
  for "is the object being found at all."
- **mAP50-95** - averaged over stricter IoU thresholds. A big gap between
  mAP50 and mAP50-95 (e.g. 0.85 vs 0.45) means boxes are found but poorly
  localized - often a labeling-precision issue, not a data-volume issue.
- **Precision vs. recall** - low recall, high precision means the model is
  too conservative (often too few positive examples); low precision, high
  recall means it's over-firing, often on a visually similar background class.

## Confusion-matrix patterns -> likely cause
| Pattern | Likely cause | Suggested next action |
|---|---|---|
| One class near-zero recall, others fine | Too few images for that class, or annotator never saw it | Re-source or re-annotate that class specifically |
| Two classes consistently confused with each other | Class confusability (visually similar) - not a data-volume problem | More *contrastive* examples of the pair, or reconsider whether they should be merged/split differently; adding generic volume won't fix this |
| High mAP50, low mAP50-95 broadly | Label/box-precision noise (loose boxes, inconsistent conventions) | Re-annotate a sample and check box tightness/consistency before adding more data |
| Broad underperformance across most classes | Model too small for the class count/complexity | Consult yolo-model-selection - consider one size up before more data |
| Good train metrics, poor val/test | Overfitting - dataset too small overall, or train/val leakage (near-duplicates across splits) | Check split hygiene (see cv-dataset-curation) before assuming more raw data is needed |

## Writing weak_classes.json
For each underperforming class, record: class name, the metric that's weak,
its sample count, and the single most likely cause from the table above -
not just the number. The orchestrator uses this cause (not the metric alone)
to decide whether to route back to sourcing, annotation, or straight to
retraining with no new data - a vague "class X has low mAP" forces the
orchestrator to guess.

## Second-pass note
If the same class is still weak after a full re-sourcing/re-annotation round,
that's evidence the cause was misdiagnosed the first time - re-examine
failure crops rather than repeating the same fix.
