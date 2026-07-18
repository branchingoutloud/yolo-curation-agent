# Dataset Plan

**Use case:** detect forklifts and pedestrians in warehouse safety camera footage to flag near‑misses

**Deployment target:** on-prem server with mid-range GPU

## Per-class image budget

| Class | Variability | Images/class | Rationale |
|---|---|---|---|
| forklift | moderate | 500 | standard forklift shape, limited pose diversity |
| pedestrian | high | 1500 | wide pose and lighting variation |

**Split ratios:** train=0.7, val=0.2, test=0.1

**Notes:** Near‑real‑time inference, target YOLOv8‑tiny or -nano for speed

