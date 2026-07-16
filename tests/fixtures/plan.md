# Dataset Plan — Traffic Monitoring (mocked planning-agent output)

> Fixture standing in for `planning-agent` so the sourcing subagent can be
> built and tested in isolation. Mirrors what planning-agent is prompted to
> write to `/workspace/plan.md`.

## Use case

Detect `car`, `truck`, `bus` from a fixed roadside camera. Deployment target
is an edge GPU (Jetson Orin), so dataset size should favor quality over raw
volume.

## Per-class image budget

| Class | Target images | Intra-class variability | Rationale |
|---|---|---|---|
| car | 500 | high | Many makes/colors/angles, day+night — needs the most coverage |
| truck | 300 | medium | Fewer body shapes, mostly highway viewing angles |
| bus | 200 | low | Consistent shape/livery — fewest images needed |

## Split

70% train / 20% val / 10% test, stratified per class.

## Sourcing guidance

Prefer already-annotated datasets (YOLO format ideally), permissive licenses
(CC0 / CC BY). Machine-readable budget is in `class_budget.json`.
