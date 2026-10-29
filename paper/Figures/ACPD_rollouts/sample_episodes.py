"""Select one evaluation episode per view and suite without outcome filtering."""

import json
from pathlib import Path
import random

from libero.libero import benchmark


def main() -> None:
    """Draw from all 10 tasks and 50 episodes using a recorded fixed seed."""
    seed = 20261029
    generator = random.Random(seed)
    suites = {
        "spatial": "libero_spatial",
        "object": "libero_object",
        "goal": "libero_goal",
        "long": "libero_10",
    }
    benchmarks = {name: benchmark.get_benchmark_dict()[key]() for name, key in suites.items()}
    selection = []
    for view in ["back", "left", "right", "top"]:
        for suite, tasks in benchmarks.items():
            task_id, episode = divmod(generator.randrange(tasks.n_tasks * 50), 50)
            task = tasks.get_task(task_id)
            record = {
                "view": view,
                "suite": suite,
                "task_id": task_id,
                "episode": episode,
                "task_name": task.name,
                "instruction": task.language,
            }
            selection.append(record)
            print(view, suite, task_id, episode, task.language)
    output = {
        "seed": seed,
        "method": "Uniform independent draw from 10 tasks x 50 evaluation initial-state episodes per view-suite cell; no outcome filtering or rerolls.",
        "selection": selection,
    }
    (Path(__file__).parent / "selection.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
