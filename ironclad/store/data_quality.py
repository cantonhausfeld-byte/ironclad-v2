"""Data quality validation for the ironclad-v2 feature store."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Critical columns checked for null rates in gold.team_game_features.
# NOTE: pass_rate_projected is a runtime model key, not a stored column.
# The stored equivalent is off_pass_rate_l4.
_NULL_RATE_CHECKS: list[tuple[str, str]] = [
    ("gold.team_game_features", "off_epa_per_play_l4"),
    ("gold.team_game_features", "implied_total_from_odds"),
    ("gold.team_game_features", "off_pass_rate_l4"),
]

_EPA_RANGE_COLS = ["epa_per_play", "epa_pass", "epa_rush"]
_EPA_MIN = -1.5
_EPA_MAX = 1.5

_GOLD_METADATA_COLS = {
    "game_id", "season", "week", "team", "opponent", "is_home",
    "cutoff_ts", "feature_version", "data_completeness_score",
}

_ROW_COUNT_TABLES = [
    "bronze.schedules",
    "silver.team_game_stats",
    "gold.team_game_features",
    "gold.player_game_features",
    "gold.betting_edges",
]

# Bronze sources required to resolve player position / starter depth. When empty,
# player features silently fall back to UNK positions and prior-only volume.
_REQUIRED_BRONZE = ["bronze.rosters", "bronze.depth_charts"]


def check_required_sources(conn, season: int) -> dict[str, Any]:
    """Count rows in bronze sources critical to position/starter resolution.

    Returns {counts: {table: int}, passed: bool}. passed is True only if all
    required sources have at least one row for the season.
    """
    counts: dict[str, int] = {}
    for table in _REQUIRED_BRONZE:
        try:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE season = ?", [season]
            ).fetchone()[0]
            counts[table] = int(n)
        except Exception as exc:
            logger.warning("Source count failed for %s: %s", table, exc)
            counts[table] = -1
    passed = all(v > 0 for v in counts.values())
    return {"counts": counts, "passed": passed}


def check_row_counts(conn, season: int) -> dict[str, Any]:
    """Count rows per key table for the given season.

    gold.betting_edges has no season column and is counted globally.
    Returns {counts: {table: int}, passed: bool}.
    """
    counts: dict[str, int] = {}
    for table in _ROW_COUNT_TABLES:
        try:
            if table == "gold.betting_edges":
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            else:
                n = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE season = ?", [season]
                ).fetchone()[0]
            counts[table] = int(n)
        except Exception as exc:
            logger.warning("Row count failed for %s: %s", table, exc)
            counts[table] = -1
    passed = all(v > 0 for v in counts.values())
    return {"counts": counts, "passed": passed}


def check_null_rates(conn, season: int) -> dict[str, Any]:
    """Check null rate for critical feature columns.

    Returns {null_rates: {"table.column": float 0–1}, passed: bool}.
    passed is True if all null rates are < 1.0 (not entirely null).
    """
    null_rates: dict[str, float] = {}
    for table, col in _NULL_RATE_CHECKS:
        key = f"{table}.{col}"
        try:
            row = conn.execute(
                f"""
                SELECT CAST(COUNT(*) - COUNT({col}) AS FLOAT)
                       / NULLIF(COUNT(*), 0)
                FROM {table}
                WHERE season = ?
                """,
                [season],
            ).fetchone()
            null_rates[key] = float(row[0]) if row[0] is not None else 1.0
        except Exception as exc:
            logger.warning("Null rate check failed for %s: %s", key, exc)
            null_rates[key] = 1.0
    passed = all(v < 1.0 for v in null_rates.values())
    return {"null_rates": null_rates, "passed": passed}


def check_epa_range(conn, season: int) -> dict[str, Any]:
    """Count rows in silver.team_game_stats with EPA outside [-1.5, 1.5].

    Returns {out_of_range: {"silver.team_game_stats": int}, passed: bool}.
    """
    table = "silver.team_game_stats"
    out_of_range: dict[str, int] = {}
    try:
        range_filter = " OR ".join(
            f"({c} IS NOT NULL AND ({c} < {_EPA_MIN} OR {c} > {_EPA_MAX}))"
            for c in _EPA_RANGE_COLS
        )
        n = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE season = ? AND ({range_filter})",
            [season],
        ).fetchone()[0]
        out_of_range[table] = int(n)
    except Exception as exc:
        logger.warning("EPA range check failed: %s", exc)
        out_of_range[table] = -1
    passed = all(v == 0 for v in out_of_range.values())
    return {"out_of_range": out_of_range, "passed": passed}


def check_completeness_score(conn, season: int) -> float:
    """Mean non-null rate across all FLOAT feature columns in gold.team_game_features.

    Columns are discovered dynamically via information_schema.
    Target: >= 0.70. Returns 0.0 if the table is empty or on error.
    """
    table = "gold.team_game_features"
    try:
        col_rows = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'gold'
              AND table_name = 'team_game_features'
              AND data_type = 'FLOAT'
            ORDER BY column_name
            """
        ).fetchall()
        float_cols = [r[0] for r in col_rows if r[0] not in _GOLD_METADATA_COLS]
        if not float_cols:
            return 0.0
        count_exprs = " + ".join(f"COUNT({c})" for c in float_cols)
        row = conn.execute(
            f"""
            SELECT CAST(({count_exprs}) AS FLOAT)
                   / NULLIF({len(float_cols)} * COUNT(*), 0)
            FROM {table}
            WHERE season = ?
            """,
            [season],
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0
    except Exception as exc:
        logger.warning("Completeness score failed: %s", exc)
        return 0.0


def run_all_checks(conn, season: int) -> dict[str, Any]:
    """Run all data quality checks and return a structured result.

    Returns a dict with sub-check results and a top-level passed: bool.
    """
    row_counts = check_row_counts(conn, season)
    null_rates = check_null_rates(conn, season)
    epa_range = check_epa_range(conn, season)
    completeness = check_completeness_score(conn, season)
    completeness_passed = completeness >= 0.70
    all_passed = (
        row_counts["passed"]
        and null_rates["passed"]
        and epa_range["passed"]
        and completeness_passed
    )
    sources = check_required_sources(conn, season)
    return {
        "season": season,
        "row_counts": row_counts,
        "null_rates": null_rates,
        "epa_range": epa_range,
        "completeness_score": completeness,
        "completeness_passed": completeness_passed,
        "sources": sources,
        "passed": all_passed and sources["passed"],
    }


def log_quality_report(conn, season: int) -> dict[str, Any]:
    """Run all checks and log a readable summary. Never raises.

    Emits loud WARNINGs for the failure modes that silently degrade forecasts
    (empty roster/depth-chart sources, low gold completeness) so they surface
    during backfill/weekly runs instead of only via the manual `validate` command.
    Returns the full results dict from run_all_checks.
    """
    results = run_all_checks(conn, season)
    sources = results["sources"]

    for table, n in sources["counts"].items():
        if n <= 0:
            logger.warning(
                "DATA QUALITY: %s is EMPTY for season %d — player positions/starters "
                "will fall back to UNK and prior-only volume. Re-run backfill or check "
                "the upstream source.",
                table, season,
            )
    if not results["completeness_passed"]:
        logger.warning(
            "DATA QUALITY: gold feature completeness %.1f%% < 70%% for season %d — "
            "features lean heavily on league priors.",
            results["completeness_score"] * 100, season,
        )
    if results["passed"]:
        logger.info(
            "Data quality OK for season %d (completeness %.1f%%)",
            season, results["completeness_score"] * 100,
        )
    else:
        logger.warning(
            "Data quality checks FAILED for season %d — see warnings above; "
            "run `ironclad validate --season %d` for the full report.",
            season, season,
        )
    return results
