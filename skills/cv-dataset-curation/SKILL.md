---
name: cv-dataset-curation
description: Estimate images-per-class, assess class imbalance, choose
  train/val/test splits, and run dedup/quality checks for an object-detection
  dataset. Use before/while sourcing data and when assembling the final
  dataset.
license: MIT
---

# CV dataset curation

## Images-per-class by intra-class variability
| Variability | Example | Rough images/class floor |
|---|---|---|
| Very low (fixed appearance) | A single logo, a fixed part number | 100-200 |
| Moderate (some pose/lighting variation) | Vehicles, packaged products | 300-500 |
| High (pose, lighting, occlusion, background) | Pedestrians, animals, generic "person" | 800-1500+ |
| Very high (fine-grained, many visually similar sub-classes) | Species/breed-level classes | 1500+ per class, and expect confusion between neighbors regardless |

Treat these as floors, not targets - more is rarely wasted, but do not
propose a plan below the floor for the class's variability tier.

## Class imbalance
- Flag any class with fewer than 30% of the median class's image count as
  "thin" - route it back to sourcing/annotation before training, don't just
  let the loss function absorb it.
- A 3:1 majority:minority ratio is usually tolerable with standard loss; past
  ~10:1, recommend either more minority-class sourcing or class-weighted
  loss/oversampling in training-agent's run.

## Split ratios
- Default: 70/20/10 (train/val/test) for datasets under ~2k images total.
- 80/10/10 once total images exceed ~5k - val doesn't need to grow linearly.
- Always split by *source image*, never by crop/augmentation - near-duplicate
  augmented versions of the same photo must not leak across splits.

## Dedup / quality checks before finalizing a split
1. Perceptual-hash or embedding-similarity dedup within each class - drop
   near-identical frames (e.g. consecutive video frames) rather than let them
   inflate an apparent image count.
2. Check that annotation coverage (bounding boxes present) is above ~95% for
   any source flagged "annotated" in sources.json - partial coverage should
   be flagged for annotation-agent, not silently included.
3. Spot-check aspect ratio / resolution outliers - extreme crops/resizes
   distort a class's effective difficulty relative to its peers.

## When re-invoked after a weak eval result
If eval-agent's diagnosis for a class is "too few images" or "high visual
variability", re-check this skill's table for that class's variability tier
before deciding how many additional images to request - don't default to an
arbitrary "add 200 more" without grounding it in the tier.
