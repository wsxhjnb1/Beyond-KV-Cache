"""Run the maintained quantization-aware trainer."""

from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("beyond.train.cli", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
