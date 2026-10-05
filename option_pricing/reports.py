"""Markdown and PDF report generation."""

from __future__ import annotations

import io
import textwrap
from datetime import datetime
from typing import Dict, List

import numpy as np

def make_markdown_report(params: Dict[str, Any], prices: Dict[str, float], greeks: Dict[str, float]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Option Pricing Laboratory Report",
        "",
        f"Generated: {now}",
        "",
        "## Parameters",
        "",
    ]
    for k, v in params.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Prices", ""]
    for k, v in prices.items():
        lines.append(f"- **{k}**: {v:.8f}" if np.isfinite(v) else f"- **{k}**: NA")
    lines += ["", "## Greeks", ""]
    for k, v in greeks.items():
        if k in {"price", "d1", "d2"}:
            continue
        lines.append(f"- **{k}**: {v:.8f}")
    lines += [
        "",
        "## Core PDE",
        "",
        "The Black-Scholes-Merton PDE is",
        "",
        r"```text",
        r"dV/dt + (r-q) S dV/dS + 0.5 sigma^2 S^2 d2V/dS2 - rV = 0",
        r"```",
        "",
        "Finite-difference schemes solve the PDE on a discrete grid. The Crank-Nicolson method uses a half-implicit, half-explicit time step and is generally more accurate and stable than the purely explicit scheme.",
    ]
    return "\n".join(lines)


def minimal_pdf_bytes(title: str, body_lines: List[str]) -> bytes:
    """Create a small text-only PDF without external dependencies."""
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_lines = ["BT", "/F1 18 Tf", "72 770 Td", f"({esc(title)}) Tj", "/F1 10 Tf", "0 -24 Td"]
    count = 0
    for raw in body_lines:
        line = raw.strip()
        if not line:
            content_lines.append("0 -12 Td")
        else:
            wrapped = textwrap.wrap(line, width=92) or [""]
            for w in wrapped:
                if count > 55:
                    break
                content_lines.append(f"({esc(w)}) Tj")
                content_lines.append("0 -13 Td")
                count += 1
        if count > 55:
            break
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objs = []
    objs.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objs.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objs.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n")
    objs.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objs.append(b"5 0 obj << /Length " + str(len(stream)).encode() + b" >> stream\n" + stream + b"\nendstream endobj\n")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(out.tell())
        out.write(obj)
    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objs)+1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer << /Root 1 0 R /Size {len(objs)+1} >>\nstartxref\n{xref_pos}\n%%EOF".encode())
    return out.getvalue()
