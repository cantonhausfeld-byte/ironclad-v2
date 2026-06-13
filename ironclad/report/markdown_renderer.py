"""Render matchup report as Markdown."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def render_markdown(context: dict, output_path: Path) -> Path:
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))
    template = env.get_template("matchup_report.md.j2")
    text = template.render(**context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def render_markdown_str(context: dict) -> str:
    """Render the matchup report template to a string (no file write)."""
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))
    return env.get_template("matchup_report.md.j2").render(**context)
