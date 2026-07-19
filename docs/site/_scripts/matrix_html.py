"""Shared HTML building blocks for the artifact-style parity matrices.

Consumed by scripts/gtap/gen_coverage_doc.py, scripts/pep/gen_pep_coverage_doc.py
(committed golden .md files, sync-tested in CI) and by render_benchmarks.py at
build time. Styling lives in _static/matrix.css (loaded site-wide via conf.py's
html_css_files); this module only emits markup with the mx-* classes.
"""
from __future__ import annotations

_TONES = {"good", "warn", "bad", "neutral"}


def raw(html: str) -> str:
    """Wrap html in a MyST raw fence so it passes through Sphinx untouched."""
    return "```{raw} html\n" + html + "\n```"


def chip(label: str, tone: str) -> str:
    assert tone in _TONES, tone
    return f'<span class="mx-chip mx-{tone}">{label}</span>'


def num(value: str, tone: str) -> str:
    assert tone in _TONES, tone
    return f'<span class="mx-num mx-{tone}">{value}</span>'


def cell(*parts: str) -> str:
    return '<div class="mx-cell">' + "".join(parts) + "</div>"


def label(ds: str, sub: str = "") -> str:
    s = f'<span class="mx-ds">{ds}</span>'
    if sub:
        s += f'<span class="mx-sub">{sub}</span>'
    return s


def ref(text: str) -> str:
    return f'<span class="mx-ref">{text}</span>'


def tablecard(headers: list[str], rows: list[list[str]]) -> str:
    """First column is the left-aligned label column; the rest are centered."""
    thead = "".join(
        f'<th class="mx-lbl">{h}</th>' if i == 0 else f"<th>{h}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for r in rows:
        tds = "".join(
            f'<td class="mx-lbl">{c}</td>' if i == 0 else f"<td>{c}</td>"
            for i, c in enumerate(r)
        )
        body.append(f"<tr>{tds}</tr>")
    return (
        '<div class="mx-card"><div class="mx-scroll"><table class="mx-table">'
        f"<thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody>"
        "</table></div></div>"
    )


def note(html: str) -> str:
    return f'<div class="mx-note"><span>⤷</span><span>{html}</span></div>'


def legend(items_html: str) -> str:
    return f'<div class="mx-legend">{items_html}</div>'


def floor_tone(f: float) -> str:
    return "good" if f >= 99 else "warn" if f >= 95 else "bad"
