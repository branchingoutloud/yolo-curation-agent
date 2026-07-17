# Training/annotation sandbox image.
#
# CPU-only by default (matches a laptop dev machine with no NVIDIA GPU - see
# README). For a CUDA machine, swap the base image for
# "nvcr.io/nvidia/pytorch:24.xx-py3" or "ultralytics/ultralytics:latest" and
# run the container with `docker run --gpus all` / add `deploy.resources` in
# docker-compose.yml.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    ultralytics \
    supervision

WORKDIR /workspace

# Keep the container alive so deepagents' execute() can `docker exec` into it
# repeatedly across a whole training run (and eval-agent reusing it
# afterward) instead of paying container-start latency per command.
CMD ["sleep", "infinity"]
