"""Discord webhook notifications for ironclad-v2."""
from __future__ import annotations

import logging

import requests

from ironclad import config

logger = logging.getLogger(__name__)

_GREEN  = 3066993
_YELLOW = 16776960
_BLUE   = 3447003


def post_run_summary(result: dict, conn=None) -> None:
    """POST weekly auto-run summary to Discord. No-ops if URL not configured."""
    url = config.DISCORD_WEBHOOK_URL
    if not url:
        logger.debug("DISCORD_WEBHOOK_URL not set; skipping notification")
        return

    s, w = result["season"], result["week"]
    gs, gt = result["games_simulated"], result["games_total"]
    edges = result["edges_saved"]
    color = _GREEN if not result["skipped"] else _YELLOW

    fields = [
        {"name": "Season / Week",   "value": f"{s} / {w}", "inline": True},
        {"name": "Games simulated", "value": f"{gs}/{gt}",  "inline": True},
        {"name": "Edges saved",     "value": str(edges),    "inline": True},
    ]

    if conn is not None and edges > 0:
        try:
            top = conn.execute("""
                SELECT player_name, stat_type, side, market_line, ev
                FROM gold.betting_edges
                ORDER BY analyzed_at DESC, ev DESC
                LIMIT 5
            """).df()
            if not top.empty:
                lines = []
                for _, r in top.iterrows():
                    line = f"**{r['player_name']}** {r['stat_type']} {r['side']}"
                    if r["market_line"] is not None:
                        line += f" {r['market_line']}"
                    line += f"  EV {r['ev']:+.3f}"
                    lines.append(line)
                fields.append({
                    "name": "Top edges",
                    "value": "\n".join(lines),
                    "inline": False,
                })
        except Exception as exc:
            logger.debug("Top-edges query failed: %s", exc)

    if result["skipped"]:
        fields.append({
            "name": f"Skipped ({len(result['skipped'])})",
            "value": "\n".join(result["skipped"][:5]),
            "inline": False,
        })

    payload = {
        "embeds": [{
            "title": f"\U0001f3c8 ironclad — {s} Week {w} complete",
            "color": color,
            "fields": fields,
        }]
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Discord notification sent")
    except Exception as exc:
        logger.warning("Discord notification failed (non-fatal): %s", exc)


def post_backfill_summary(result: dict) -> None:
    """POST post-game backfill summary to Discord. No-ops if URL not configured."""
    url = config.DISCORD_WEBHOOK_URL
    if not url:
        logger.debug("DISCORD_WEBHOOK_URL not set; skipping notification")
        return

    ingest = result.get("ingest", {})
    silver = result.get("silver", {})
    fb = result.get("features_built", 0)
    ff = result.get("features_failed", 0)
    color = _GREEN if ff == 0 else _YELLOW

    fields = [
        {"name": "PBP rows",          "value": str(ingest.get("play_by_play", "?")),  "inline": True},
        {"name": "Roster rows",        "value": str(ingest.get("rosters", "?")),       "inline": True},
        {"name": "Features built",     "value": f"{fb}" + (f" ({ff} failed)" if ff else ""), "inline": True},
        {"name": "Silver games",       "value": str(silver.get("games", "?")),         "inline": True},
        {"name": "Silver player rows", "value": str(silver.get("player_game_stats", "?")), "inline": True},
    ]

    payload = {
        "embeds": [{
            "title": "\U0001f4ca ironclad — post-game backfill complete",
            "color": color,
            "fields": fields,
        }]
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Discord backfill notification sent")
    except Exception as exc:
        logger.warning("Discord notification failed (non-fatal): %s", exc)


def post_retrain_summary(result: dict) -> None:
    """POST end-of-season retrain summary to Discord. No-ops if URL not configured."""
    url = config.DISCORD_WEBHOOK_URL
    if not url:
        logger.debug("DISCORD_WEBHOOK_URL not set; skipping notification")
        return

    train = result.get("train", {})
    fields = []
    for model_name, metrics in train.items():
        val = metrics.get("val_brier") or metrics.get("val_log_loss") or metrics.get("accuracy")
        val_str = f"{val:.4f}" if isinstance(val, float) else "—"
        fields.append({"name": model_name, "value": val_str, "inline": True})

    payload = {
        "embeds": [{
            "title": "\U0001f3c6 ironclad — end-of-season retrain complete",
            "color": _GREEN,
            "fields": fields if fields else [{"name": "Models", "value": ", ".join(train.keys()) or "none", "inline": False}],
        }]
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Discord retrain notification sent")
    except Exception as exc:
        logger.warning("Discord notification failed (non-fatal): %s", exc)


def _fence_tables(text: str) -> str:
    """Wrap contiguous pipe-table blocks in triple-backtick fences for Discord monospace."""
    lines = text.splitlines()
    out: list[str] = []
    in_table = False
    for line in lines:
        is_table_line = line.startswith("|")
        if is_table_line and not in_table:
            out.append("```")
            in_table = True
        elif not is_table_line and in_table:
            out.append("```")
            in_table = False
        out.append(line)
    if in_table:
        out.append("```")
    return "\n".join(out)


def _split_by_subsection(section: str, limit: int) -> list[str]:
    """Split an oversized section on `###` boundaries, re-closing fences at breaks."""
    if len(section) <= limit:
        return [section]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in section.splitlines(keepends=True):
        if line.lstrip().startswith("### ") and current_len + len(line) > limit and current:
            parts.append("".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)
    if current:
        parts.append("".join(current))
    return parts


def _md_chunks(text: str, limit: int = 1900) -> list[str]:
    """Split a matchup report on `---` section separators, fence tables, then greedily pack."""
    sections = text.split("\n---\n")
    fenced = [_fence_tables(s.strip()) for s in sections if s.strip()]

    expanded: list[str] = []
    for sec in fenced:
        expanded.extend(_split_by_subsection(sec, limit))

    chunks: list[str] = []
    current = ""
    for sec in expanded:
        candidate = sec if not current else f"{current}\n\n---\n\n{sec}"
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sec
    if current:
        chunks.append(current)
    return chunks


def post_model_output(
    game_id: str,
    report_md: str,
    top_edges_df=None,
    win_prob_home: float | None = None,
    win_prob_away: float | None = None,
    home_team: str = "",
    away_team: str = "",
    projected_home: int | None = None,
    projected_away: int | None = None,
    season: int | None = None,
    week: int | None = None,
) -> None:
    """POST a matchup report to Discord: summary embed then full report as inline messages."""
    url = config.DISCORD_WEBHOOK_URL
    if not url:
        logger.debug("DISCORD_WEBHOOK_URL not set; skipping model output notification")
        return

    title = f"\U0001f3c8 {away_team} @ {home_team}"
    if season and week:
        title += f" — {season} Week {week}"

    fields = []
    if win_prob_home is not None and win_prob_away is not None:
        fields.append({
            "name": f"{home_team} win prob",
            "value": f"**{win_prob_home * 100:.1f}%**",
            "inline": True,
        })
        fields.append({
            "name": f"{away_team} win prob",
            "value": f"**{win_prob_away * 100:.1f}%**",
            "inline": True,
        })
    if projected_home is not None and projected_away is not None:
        fields.append({
            "name": "Projected score",
            "value": f"{away_team} {projected_away} — {home_team} {projected_home}",
            "inline": False,
        })

    if top_edges_df is not None and not top_edges_df.empty:
        lines = []
        for _, r in top_edges_df.head(5).iterrows():
            line_str = f"{r['market_line']}" if r.get("market_line") is not None else "—"
            ev_str = f"{r['ev']:+.3f}" if r.get("ev") is not None else ""
            lines.append(
                f"`{str(r['player_name']):<22} {str(r['stat_type']):<12} "
                f"{str(r['side']):<5} {line_str:<6} EV {ev_str}`"
            )
        if lines:
            fields.append({
                "name": f"Top edges ({len(top_edges_df)} total)",
                "value": "\n".join(lines),
                "inline": False,
            })

    embed = {
        "title": title,
        "color": _BLUE,
        "fields": fields,
        "footer": {"text": game_id},
    }

    try:
        resp = requests.post(url, json={"embeds": [embed]}, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Discord model output failed (non-fatal): %s", exc)
        return

    from ironclad.notifications.formatters import (
        format_team_comparison_block,
        format_top_edges_block,
    )

    # Team-comparison block (season-to-date side-by-side stats)
    if season and home_team and away_team:
        try:
            from ironclad.store.connection import get_connection
            block = format_team_comparison_block(
                get_connection(read_only=True), home_team, away_team, season,
            )
            if block:
                _send_content(url, f"```\n{block}\n```", game_id)
        except Exception as exc:
            logger.warning("Team-comparison block failed for %s: %s", game_id, exc)

    # Top-edges block (full top-10)
    if top_edges_df is not None and not top_edges_df.empty:
        try:
            block = format_top_edges_block(top_edges_df, limit=10)
            _send_content(url, f"```\n{block}\n```", game_id)
        except Exception as exc:
            logger.warning("Top-edges block failed for %s: %s", game_id, exc)

    # Send full report as inline chat messages (chunked to stay under Discord's 2000-char limit)
    for chunk in _md_chunks(report_md):
        if not _send_content(url, chunk, game_id):
            break

    logger.info("Discord model output posted for %s", game_id)


def _send_content(url: str, content: str, game_id: str) -> bool:
    """POST a content message; return False on error."""
    try:
        resp = requests.post(url, json={"content": content}, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Discord message failed for %s: %s", game_id, exc)
        return False
