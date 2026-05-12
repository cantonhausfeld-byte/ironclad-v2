"""Discord webhook notifications for ironclad-v2."""
from __future__ import annotations

import logging

import requests

from ironclad import config

logger = logging.getLogger(__name__)

_GREEN  = 3066993
_YELLOW = 16776960


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
