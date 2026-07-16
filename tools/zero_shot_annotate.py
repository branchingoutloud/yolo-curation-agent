"""Zero-shot open-vocabulary detector used by annotation-agent.

Runs inside the annotation sandbox via execute() - not exposed as a separate
MCP service, since it's just local inference over a batch of images. Per §13
of the architecture doc: YOLO-World is lighter than Grounding DINO if the
sandbox GPU is constrained - swap the model loaded here, not the tool's
signature or call sites.

The actual model call is intentionally unimplemented (TODO below) - wire in
whichever zero-shot detector you land on before annotation-agent's first run.
"""

from langchain_core.tools import tool


@tool
def zero_shot_annotate(image_dir: str, classes: list[str], out_dir: str) -> str:
    """Run a zero-shot open-vocabulary detector over image_dir for the given
    classes, writing YOLO-format label files to out_dir. Returns a short
    summary (image count, per-class detection count).
    """
    # TODO: load a zero-shot detector (e.g. YOLO-World or Grounding DINO),
    # run inference over every image in image_dir restricted to `classes`,
    # and write YOLO-format .txt label files into out_dir.
    raise NotImplementedError(
        "zero_shot_annotate has no detector wired in yet. "
        f"Requested image_dir={image_dir!r} classes={classes!r} out_dir={out_dir!r}"
    )
