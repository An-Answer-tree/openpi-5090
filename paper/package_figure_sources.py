"""Package editable paper figures, their sources, and selected video material."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile


def collect_files(directory: Path):
    """Yield figure assets without walking dependency or cache directories."""
    for current, directories, names in os.walk(directory):
        directories[:] = sorted(
            name for name in directories if name not in {"node_modules", "__pycache__", ".venv", ".git"}
        )
        for name in sorted(names):
            if not name.endswith(".pyc"):
                yield Path(current) / name


def main() -> None:
    """Write a ZIP with per-file hashes and original video provenance."""
    root = Path(__file__).resolve().parents[1]
    date = datetime.now().astimezone().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "tmp" / f"ACPD_figure_sources_{date}.zip")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir = root / "paper/Figures"
    files = [(source, source.relative_to(root).as_posix()) for source in collect_files(figure_dir)]
    files.append((Path(__file__).resolve(), "paper/package_figure_sources.py"))
    files.append((figure_dir / "EDITING_GUIDE.md", "README.md"))

    episode_dir = root / "tmp/libero_spatial_episode0_6view_8frames"
    if not episode_dir.exists():
        episode_dir = root / "materials/synchronized_episode"
    if episode_dir.exists():
        files.extend(
            (source, f"materials/synchronized_episode/{source.relative_to(episode_dir).as_posix()}")
            for source in collect_files(episode_dir)
        )

    rollout_manifest = json.loads((figure_dir / "ACPD_rollouts/manifest.json").read_text())
    bundled_index = root / "materials/evaluation_videos/index.json"
    previous_videos = {}
    if bundled_index.exists():
        previous_videos = {
            (record["view"], record["suite"]): record for record in json.loads(bundled_index.read_text())["videos"]
        }
    videos = []
    for record in rollout_manifest["rollouts"]:
        key = f"{record['view']}_{record['suite']}"
        source = Path(record["source_video"])
        if not source.exists():
            previous = previous_videos[(record["view"], record["suite"])]
            source = bundled_index.parent / previous["file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["source_sha256"]:
            raise ValueError(f"Video differs from recorded source: {source}")
        name = f"{key}.mp4"
        files.append((source, f"materials/evaluation_videos/{name}"))
        videos.append(
            {
                "view": record["view"],
                "suite": record["suite"],
                "file": name,
                "source_video": record["source_video"],
                "source_sha256": record["source_sha256"],
            }
        )

    git_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit_at_export": git_result.stdout.strip() if git_result.returncode == 0 else None,
        "source": "Current working-tree files; hashes identify exact exported versions.",
        "files": [],
    }
    prefix = output.stem
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:

        def add_file(name: str, data: bytes) -> None:
            archive.writestr(f"{prefix}/{name}", data)
            manifest["files"].append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

        for source, name in files:
            add_file(name, source.read_bytes())
        add_file(
            "materials/evaluation_videos/index.json",
            (json.dumps({"videos": videos}, ensure_ascii=False, indent=2) + "\n").encode(),
        )
        archive.writestr(f"{prefix}/MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n")
    print(f"Archive: {output}")
    print(f"Files: {len(manifest['files']) + 1}; size: {output.stat().st_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
