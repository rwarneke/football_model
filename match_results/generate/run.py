"""Run pull_results followed by clean_results."""

from __future__ import annotations

import importlib

from . import pull_results


def main() -> None:
    pull_results.main()
    importlib.import_module("match_results.generate.clean_results")


if __name__ == "__main__":
    main()
