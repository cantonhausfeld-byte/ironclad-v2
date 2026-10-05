"""Weekly auto-run: every upcoming game gets a report, props or not."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from ironclad.workflow.auto_run import AutoRunWorkflow


def _add_game(conn, game_id, gameday):
    conn.execute(
        "INSERT INTO silver.games (game_id, season, season_type, week, gameday, home_team, "
        "away_team) VALUES (?, 2026, 'REG', 5, ?, 'BAL', 'TEN')",
        [game_id, gameday],
    )


def test_reports_and_posts_every_game_without_props(conn, tmp_path):
    soon = (date.today() + timedelta(days=3)).isoformat()
    _add_game(conn, "2026_05_TEN_BAL", soon)
    _add_game(conn, "2026_05_KC_LV", soon)

    result = MagicMock(home_team="BAL", away_team="TEN")
    result.win_probability.return_value = (0.7, 0.3)
    result.score_summary.return_value = {"home_score_mean": 27.4, "away_score_mean": 18.2}
    cutoff = datetime(2026, 10, 11, 16, 30, tzinfo=timezone.utc)

    sim = MagicMock(side_effect=lambda gid: (result, {"game_id": gid, "season": 2026, "week": 5},
                                             cutoff, pd.DataFrame()))
    write = MagicMock(side_effect=lambda r, meta, c, out, fmt: (out / f"{meta['game_id']}.md", {}))

    with (
        patch("ironclad.workflow.auto_run.get_connection", return_value=conn),
        patch("ironclad.workflow.weekly.WeeklyWorkflow") as weekly,
        patch("ironclad.ingest.injuries.InjuryIngestor"),
        patch("ironclad.workflow.matchup.MatchupWorkflow.simulate", sim),
        patch("ironclad.workflow.matchup.MatchupWorkflow.write_report", write),
        patch("ironclad.betting.props.load_prop_lines_from_db", return_value=[]),
        patch("ironclad.config.DISCORD_WEBHOOK_URL", "https://example.invalid/hook"),
        patch("ironclad.notifications.discord.post_model_output") as post,
        patch("ironclad.report.markdown_renderer.render_markdown_str", return_value="# report"),
    ):
        weekly.return_value.run.return_value = {}
        out = AutoRunWorkflow(skip_odds=True, output_dir=tmp_path).run(season=2026, week=5)

    assert out["games_simulated"] == 2
    assert out["reports_written"] == 2
    assert out["edges_saved"] == 0
    assert write.call_count == 2
    assert {c.kwargs["game_id"] for c in post.call_args_list} == {"2026_05_TEN_BAL", "2026_05_KC_LV"}
    assert all(c.kwargs["top_edges_df"] is None for c in post.call_args_list)


def test_missing_odds_key_skips_odds_instead_of_failing(conn, tmp_path):
    with (
        patch("ironclad.workflow.auto_run.get_connection", return_value=conn),
        patch("ironclad.workflow.weekly.WeeklyWorkflow") as weekly,
        patch("ironclad.ingest.injuries.InjuryIngestor"),
        patch("ironclad.config.ODDS_API_KEY", ""),
        patch("ironclad.ingest.odds.OddsIngestor") as odds,
    ):
        weekly.return_value.run.return_value = {}
        out = AutoRunWorkflow(output_dir=tmp_path).run(season=2026, week=5)

    odds.assert_not_called()
    assert out["odds_rows"] == 0
