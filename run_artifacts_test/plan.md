# Dataset Plan

**Use case:** detect forklifts and pedestrians in warehouse safety camera footage to flag near-misses

**Deployment target:** on-prem server with mid-range GPU, near-real-time inference

## Per-class image budget

| Class | Variability | Images/class | Rationale |
|---|---|---|---|
| forklift | moderate | 400 | Moderate pose and lighting variations; 400 images provide sufficient coverage. |
| pedestrian | high | 800 | High intra-class variability (poses, occlusions, lighting); 800 images ensure robust detection. |

**Split ratios:** train=0.7, val=0.2, test=0.1

**Notes:** Dataset sized for YOLOv5s/v8n model; total 1200 images balances accuracy and training time.

