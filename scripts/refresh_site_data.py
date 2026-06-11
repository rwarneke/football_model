from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.world_cup_results import PUBLIC_MODEL_OUTPUT_DIR, copy_results_wc2026_to_public

SIM_N = "10000"
PRETOURNAMENT_OUTPUT_DIR = ROOT / "web" / "public" / "model_output_pretournament"


def run_step(name: str, cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n==> {name}")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def ensure_pretournament_snapshot() -> None:
    if PRETOURNAMENT_OUTPUT_DIR.exists():
        return
    if not PUBLIC_MODEL_OUTPUT_DIR.exists():
        raise FileNotFoundError(
            "Cannot create pre-tournament snapshot because web/public/model_output does not exist."
        )
    print(f"\n==> Freeze pre-tournament public model output -> {PRETOURNAMENT_OUTPUT_DIR}")
    PRETOURNAMENT_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PUBLIC_MODEL_OUTPUT_DIR, PRETOURNAMENT_OUTPUT_DIR)


def main() -> None:
    base_env = os.environ.copy()
    sim_env = base_env.copy()
    sim_env["SIM_N"] = SIM_N

    ensure_pretournament_snapshot()
    run_step(
        "Pull and clean match results",
        [sys.executable, "-m", "match_results.generate.run", "--no-goalscorers"],
    )
    copy_results_wc2026_to_public()
    run_step("Sync reference data into web/public", [sys.executable, "scripts/sync_reference_data.py"])
    run_step("Fit model", [sys.executable, "-m", "src.fit_model"])
    run_step("Export matchup probabilities", [sys.executable, "-m", "src.export_win_probabilities"])
    run_step("Run tournament simulations", [sys.executable, "-m", "src.run_simulations"], env=sim_env)
    run_step("Postprocess simulation outputs", [sys.executable, "-m", "src.postprocess_simulations"])

    print("\nRefresh complete.")


if __name__ == "__main__":
    main()
