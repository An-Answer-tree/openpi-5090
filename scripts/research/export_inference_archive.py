"""Export selected OpenPI weights, normalization assets, and evaluation counts."""

# ruff: noqa: RUF001

import argparse
import json
import os
from pathlib import Path
import shutil

import yaml


def export_archive(manifest_path: Path, destination: Path) -> None:
    """Build a separate inference archive without modifying its source."""
    manifest = yaml.safe_load(manifest_path.read_text())
    source_root = Path(manifest["archive_root"])
    destination.mkdir()
    exported = {
        "format": "openpi_inference",
        "suite_order": manifest["suite_order"],
        "episodes_per_suite": manifest["episodes_per_suite"],
        "models": {},
    }
    table = [
        "# 模型与验证结果",
        "",
        "每套评测回合数见清单，模型目录均直接包含 `params/` 和 `assets/`。",
        "",
        "| 模型 | Spatial | Object | Goal | LIBERO-10 | 平均 | 模型目录 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, model in manifest["models"].items():
        source = source_root / model["archive"]
        relative = Path(model["archive"])
        if relative.name.isdecimal():
            relative = relative.parent
        checkpoint = destination / relative
        checkpoint.mkdir(parents=True)
        for item in ("params", "assets"):
            shutil.copytree(source / item, checkpoint / item, copy_function=os.link)
        entry = {"label": model["label"], "checkpoint": str(relative)}
        if "policy_config" in model:
            entry["policy_config"] = model["policy_config"]
        if "exact_contribution_injection" in model:
            entry["exact_contribution_injection"] = model["exact_contribution_injection"]
        if "successes" in model:
            result = {
                "suite_order": manifest["suite_order"],
                "episodes_per_suite": manifest["episodes_per_suite"],
                "successes": model["successes"],
            }
            result_dir = destination / "results" / relative
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            total = sum(model["successes"])
            episodes = manifest["episodes_per_suite"]
            rates = [count / episodes * 100 for count in model["successes"]]
            mean = total / (episodes * len(rates)) * 100
            lines = [
                f"{suite}: {count}/{episodes} ({rate:.2f}%)"
                for suite, count, rate in zip(manifest["suite_order"], model["successes"], rates, strict=True)
            ]
            lines.append(f"pooled: {total}/{episodes * len(rates)} ({mean:.2f}%)")
            (result_dir / "summary.txt").write_text("\n".join(lines) + "\n")
            entry["result_summary"] = str(Path("results") / relative / "summary.json")
            percentages = " | ".join(f"{rate:.2f}%" for rate in rates)
            table.append(f"| {model['label']} | {percentages} | {mean:.2f}% | `{relative}` |")
        else:
            table.append(f"| {model['label']} | — | — | — | — | — | `{relative}` |")
        exported["models"][name] = entry

    (destination / "curated-checkpoints.yaml").write_text(yaml.safe_dump(exported, allow_unicode=True, sort_keys=False))
    (destination / "FINAL_RESULTS.md").write_text("\n".join(table) + "\n")
    (destination / "README.md").write_text(
        "# ACPD 推理模型归档\n\n"
        "模型与加载配置见 `curated-checkpoints.yaml`；验证结果见 `FINAL_RESULTS.md`。\n"
        "每个模型仅包含 `params/` 权重和 `assets/` 归一化参数，用于推理。\n"
        "推理时使用清单中的 policy_config 和 checkpoint 路径；teacher 保留完整权重。\n"
        "ACL-only 与关闭注入的模型必须使用清单指定的关闭注入配置。\n"
        "关闭注入一项使用完整 ACPD 训练权重，仅在推理时关闭注入。\n"
        "MV-SV-KD 为 pi0.5 上的适配版本。\n"
        "本目录不含优化器或训练状态，不能直接断点续训。\n"
        "各方法的训练协议不同，归档结果不代表相同训练预算的受控比较。\n"
    )


def main() -> None:
    """Parse the source selection and the new archive destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_archive(args.manifest, args.destination)


if __name__ == "__main__":
    main()
