"""Extract unmodified frames from archived ACPD evaluation videos."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

SUITE_DIRS = {"spatial": "spatial", "object": "object", "goal": "goal", "long": "libero_10"}


def verify_outcome(log: Path, instruction: str, episode: int, result: str) -> None:
    """Check the video's outcome against its original evaluation log."""
    active = False
    with log.open(errors="replace") as stream:
        for line in stream:
            if "Task:" in line and "episode:" in line:
                active = f"Task: {instruction}; episode: {episode}" in line
            if active and "Success:" in line:
                logged = "success" if "Success: True" in line else "failure"
                if logged != result:
                    raise ValueError(f"Video/log outcome mismatch: {log}, episode {episode}")
                return
    raise ValueError(f"Selected episode outcome not found: {log}, episode {episode}")


def main() -> None:
    """Export six frames per rollout and record their exact provenance."""
    base = Path(__file__).resolve().parent
    repository = next(p for p in base.parents if (p / "experiments/curated-checkpoints.yaml").exists())
    model_index = yaml.safe_load((repository / "experiments/curated-checkpoints.yaml").read_text())
    selection = json.loads((base / "selection.json").read_text())
    records = []
    for task in selection["selection"]:
        view, suite = task["view"], task["suite"]
        model = model_index["models"][f"student_{view}view"]
        root = Path(model["result_summary"]).parent
        stem = task["instruction"].replace(" ", "_") + f"_{task['episode']:02d}"
        candidates = [root / SUITE_DIRS[suite] / "videos" / f"{stem}_{result}.mp4" for result in ["success", "failure"]]
        source = next((p for p in candidates if p.exists()), None)
        if source is None:
            raise FileNotFoundError(f"Selected episode is missing; selection was not changed: {candidates}")
        result = source.stem.rsplit("_", 1)[1]
        log = root / "logs" / f"{SUITE_DIRS[suite]}.log"
        verify_outcome(log, task["instruction"], task["episode"], result)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open evaluation video: {source}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        indices = np.linspace(0, frame_count - 1, 6).round().astype(int).tolist()
        destination = base / "frames" / f"{view}_{suite}"
        destination.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index}: {source}")
            target = destination / f"frame_{index:03d}.png"
            cv2.imwrite(str(target), frame)
            frames.append({"index": index, "path": str(target.relative_to(base))})
        capture.release()
        record = {
            **task,
            "result": result,
            "source_video": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model_archive": model["archive"],
            "policy_config": model["policy_config"],
            "evaluation_summary": str(root / "summary.txt"),
            "evaluation_log": str(log),
            "log_outcome_verified": True,
            "frame_count": frame_count,
            "video_fps": fps,
            "frame_size": [frame.shape[1], frame.shape[0]],
            "frames": frames,
        }
        records.append(record)
        print(view, suite, record["result"], frame_count, indices)
    manifest = {
        "method": "ACPD",
        "selection": selection["method"],
        "selection_seed": selection["seed"],
        "frame_selection": "Six uniformly spaced recorded frames, including the first and last recorded frames.",
        "pixels": "Original decoded RGB frames; no cropping, enhancement, retouching, or generated content.",
        "step_labels": "Zero-based indices within the recorded policy rollout, after the evaluator's initial settling steps. Playback fps does not specify physical simulation time.",
        "outcome_source": "Success/failure labels are read from the evaluator's original video filenames; the final displayed image is the last recorded pre-action frame.",
        "rollouts": records,
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
