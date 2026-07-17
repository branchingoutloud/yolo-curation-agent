# Eval report

- Split evaluated: `val`
- Overall mAP50: **0.0000**
- Overall mAP50-95: **0.0000**
- Precision: 0.0000  |  Recall: 0.0000

## Per-class

| class | mAP50 | mAP50-95 | train boxes | val boxes |
|---|---|---|---|---|
| car | 0.000 | 0.000 | 1 | 1 |
| truck | 0.000 | 0.000 | 2 | 0 |
| bus | 0.000 | 0.000 | 0 | 0 |

## Weak classes

- **car** (mAP50 0.0): low mAP50 - check label quality / add harder or more varied examples
- **truck** (mAP50 0.0): low mAP50 - check label quality / add harder or more varied examples
- **bus** (mAP50 0.0): under-represented in training (0 boxes vs ~2 median); low mAP50 - check label quality / add harder or more varied examples

## Confusion matrix

Raw ultralytics matrix (last row/col = background). Large off-diagonal entries between two classes indicate they are being confused for each other.

```json
[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
```
