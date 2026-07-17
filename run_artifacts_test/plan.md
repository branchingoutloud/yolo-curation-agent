# Dataset Plan

**Use case:** Detect forklifts and pedestrians to flag near‑misses in warehouse safety camera footage.

**Deployment target:** on‑prem server with a mid‑range GPU

## Per-class image budget

| Class | Variability | Images/class | Rationale |
|---|---|---|---|
| forklift | high | 2000 | High pose, lighting, and occlusion variability in warehouse scenes; meets high‑tier floor. |
| pedestrian | very_high | 4000 | Extremely varied poses, clothing, and backgrounds; meets very‑high‑tier floor. |

**Split ratios:** train=0.7, val=0.2, test=0.1

**Notes:** Near real‑time detection for warehouse safety. Balanced representation to avoid bias.

