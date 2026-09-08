import logging

import tyro

from examples.libero.multiview_eval.evaluator import EvalArgs
from examples.libero.multiview_eval.evaluator import MultiviewLiberoEvaluator


def main(args: EvalArgs) -> None:
    MultiviewLiberoEvaluator(args).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(EvalArgs))
