"""Export simulation results to CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ironclad.simulation.results import SimulationResult


def export_player_csv(result: SimulationResult, output_path: Path) -> Path:
    player_df = result.player_summary()
    if player_df.empty:
        output_path.write_text("No player data\n")
        return output_path

    metrics = ["targets", "receptions", "rec_yards", "carries", "rush_yards",
               "pass_attempts", "completions", "pass_yards", "tds"]
    id_cols = ["player_id", "player_name", "team", "position", "is_home"]

    pivot_rows = []
    for (pid, pname, team, pos, is_home), grp in player_df.groupby(id_cols):
        row = {"player_id": pid, "player_name": pname, "team": team, "position": pos}
        for m in metrics:
            mdf = grp[grp["metric"] == m]
            if not mdf.empty:
                row[f"{m}_p25"] = round(mdf.iloc[0]["p25"], 1)
                row[f"{m}_p50"] = round(mdf.iloc[0]["p50"], 1)
                row[f"{m}_p75"] = round(mdf.iloc[0]["p75"], 1)
                row[f"{m}_mean"] = round(mdf.iloc[0]["mean"], 2)
        pivot_rows.append(row)

    df = pd.DataFrame(pivot_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def export_team_csv(result: SimulationResult, output_path: Path) -> Path:
    team_df = result.team_summary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    team_df.to_csv(output_path, index=False)
    return output_path
