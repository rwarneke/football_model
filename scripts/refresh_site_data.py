from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SIM_N = "10000"


def run_step(name: str, cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n==> {name}")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    base_env = os.environ.copy()
    sim_env = base_env.copy()
    sim_env["SIM_N"] = SIM_N

    run_step("Pull and clean match results", [sys.executable, "-m", "match_results.generate.run"])
    run_step("Sync reference data into web/public", [sys.executable, "scripts/sync_reference_data.py"])
    run_step("Fit model", [sys.executable, "-m", "src.fit_model"])
    run_step("Run tournament simulations", [sys.executable, "-m", "src.run_simulations"], env=sim_env)
    run_step("Postprocess simulation outputs", [sys.executable, "-m", "src.postprocess_simulations"])

    print("\nRefresh complete.")


if __name__ == "__main__":
    main()
