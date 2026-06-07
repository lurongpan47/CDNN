#!/usr/bin/env python3
"""Render the CD-Transformer benchmark report as a PDF *with the charts*.

Reuses benchmark_report.py's parsing, cost model, convergence analysis and
matplotlib charts, then lays everything out with reportlab. The four charts
(total-loss-vs-CE, perplexity, throughput, kappa) are embedded as images.

Usage:
  python report_to_pdf.py --log run.log --config small --seq_len 2048 \
      --world_size 8 --out report.pdf
"""
import argparse, base64, io, re
from pathlib import Path

import benchmark_report as br
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable)


# --- markdown -> reportlab inline formatting --------------------------------
def inline(md: str) -> str:
    """Convert a subset of markdown inline syntax to reportlab markup."""
    md = (md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    md = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", md)
    md = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', md)
    md = re.sub(r"(?<![\*])\*(?!\s)(.+?)(?<!\s)\*", r"<i>\1</i>", md)
    # superscript-ish: keep ASCII (reportlab core fonts lack unicode super/sub)
    md = md.replace("κ", "kappa").replace("→", "-&gt;").replace("×", "x")
    md = md.replace("≈", "~").replace("≤", "&lt;=").replace("≥", "&gt;=")
    md = md.replace("·", "*").replace("—", "-").replace("’", "'")
    return md


def build_pdf(md: str, charts: dict, out_path: str, lang: str, arch_fig: str = None):
    base_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    if lang == "zh":
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        base_font = bold_font = "STSong-Light"   # CID font has no separate bold

    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Title"], fontName=bold_font, fontSize=18, spaceAfter=4,
                        textColor=colors.HexColor("#111827"))
    SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontName=base_font, fontSize=9,
                         textColor=colors.HexColor("#6b7280"), spaceAfter=10)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=bold_font, fontSize=13,
                        textColor=colors.HexColor("#1d4ed8"), spaceBefore=12,
                        spaceAfter=4)
    BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontName=base_font, fontSize=9.5,
                          leading=14, spaceAfter=3)
    BULLET = ParagraphStyle("BUL", parent=BODY, leftIndent=12, bulletIndent=2)
    QUOTE = ParagraphStyle("Q", parent=BODY, leftIndent=10, textColor=colors.HexColor("#374151"),
                           borderColor=colors.HexColor("#d1d5db"), backColor=colors.HexColor("#f9fafb"),
                           borderPadding=6, spaceBefore=4, spaceAfter=6, fontSize=9)

    chart_titles = {
        "loss": "Total loss vs decoded cross-entropy" if lang == "en" else "总损失 vs 解码交叉熵",
        "ppl":  "Convergence — perplexity vs step" if lang == "en" else "收敛 — 困惑度 vs step",
        "toks": "Throughput (tokens/s)" if lang == "en" else "吞吐 (tokens/s)",
        "kappa":"CDLinear condition number kappa" if lang == "en" else "CDLinear 条件数 kappa",
    }
    CAP = ParagraphStyle("CAP", parent=BODY, fontSize=8,
                         textColor=colors.HexColor("#6b7280"), spaceBefore=2, spaceAfter=10)

    def chart_flowables():
        out = []
        avail = A4[0] - 36 * mm
        for key in ("loss", "ppl", "toks", "kappa"):
            if key in charts:
                img = Image(io.BytesIO(base64.b64decode(charts[key])))
                ratio = img.imageHeight / img.imageWidth
                img.drawWidth = avail
                img.drawHeight = avail * ratio
                out.append(img)
                out.append(Paragraph(chart_titles.get(key, key), CAP))
        return out

    story, in_table, tbl = [], False, []

    def flush_table():
        nonlocal in_table, tbl
        if tbl:
            data = [[Paragraph(inline(c), BODY) for c in row] for row in tbl]
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eff6ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (-1, -1), base_font),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t); story.append(Spacer(1, 6))
        in_table, tbl = False, []

    charts_inserted = False
    arch_inserted = False
    import os
    for raw in md.splitlines():
        line = raw.rstrip()
        is_tbl = line.startswith("|")
        if in_table and not is_tbl:
            flush_table()
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), H1))
        elif line.startswith("## "):
            # architecture figure goes once, just before the first section
            if not arch_inserted and arch_fig and os.path.exists(arch_fig):
                aimg = Image(arch_fig)
                avail = A4[0] - 36 * mm
                aimg.drawWidth = avail
                aimg.drawHeight = avail * (aimg.imageHeight / aimg.imageWidth)
                story.append(aimg)
                story.append(Spacer(1, 8))
                story.append(HRFlowable(width="100%", color=colors.HexColor("#e5e7eb")))
                arch_inserted = True
            # insert charts right before section 3 (cost), like the HTML report
            if not charts_inserted and line.lstrip("# ").startswith("3"):
                story += chart_flowables()
                story.append(HRFlowable(width="100%", color=colors.HexColor("#e5e7eb")))
                charts_inserted = True
            story.append(Paragraph(inline(line[3:]), H2))
        elif line.startswith("> "):
            story.append(Paragraph(inline(line[2:]), QUOTE))
        elif line.startswith("- "):
            story.append(Paragraph(inline(line[2:]), BULLET, bulletText="•"))
        elif is_tbl:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue  # markdown header separator row
            in_table = True; tbl.append(cells)
        elif line.startswith("_") and line.endswith("_") and len(line) > 2:
            story.append(Paragraph(inline(line[1:-1]), SUB))
        elif line.strip():
            story.append(Paragraph(inline(line), BODY))
        else:
            story.append(Spacer(1, 4))
    if in_table:
        flush_table()
    if not charts_inserted:           # no section 3 -> append charts at end
        story += chart_flowables()

    SimpleDocTemplate(out_path, pagesize=A4,
                      leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=16 * mm,
                      title="CD-Transformer benchmark report").build(story)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--config", default="small")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--world_size", type=int, default=8)
    ap.add_argument("--gpu_tflops", type=float, default=989.0)
    ap.add_argument("--mtp_weight", type=float, default=0.3)
    ap.add_argument("--baseline_log", default=None)
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    ap.add_argument("--arch_fig", default=None,
                    help="optional architecture figure PNG to embed as Figure 1")
    ap.add_argument("--out", default="report.pdf")
    ap.add_argument("--val_loss", type=float, default=None)
    ap.add_argument("--val_tokens", type=int, default=0)
    ap.add_argument("--train_loss", type=float, default=None)
    ap.add_argument("--train_tokens", type=int, default=0)
    args = ap.parse_args()

    parsed = br.parse_log(Path(args.log))
    arch = br.analyze_from_config(args.config, flop_tokens=args.seq_len)
    mtp_w = args.mtp_weight
    br.decode_ce(parsed, mtp_w)
    hdr = parsed.get("header") or {}
    if arch and hdr.get("total_params"):
        arch["model_params"] = hdr["total_params"]
        if hdr.get("dense_equiv_params"):
            arch["model_dense_equiv_params"] = hdr["dense_equiv_params"]
            arch["model_param_compression"] = hdr["dense_equiv_params"] / hdr["total_params"]
    tps = hdr["effective_batch"] * hdr["seq_len"] if hdr.get("effective_batch") and hdr.get("seq_len") else None
    cost = br.deepseek_baseline_cost(arch, seq_len=args.seq_len) if arch else None
    conv = br.convergence_stats(parsed, tokens_per_step=tps)
    baseline_parsed = None
    if args.baseline_log:
        baseline_parsed = br.parse_log(Path(args.baseline_log))
        br.decode_ce(baseline_parsed, mtp_w)
    charts = br.make_charts(parsed, baseline_parsed=baseline_parsed)
    import math as _m
    valres = None
    if args.val_loss is not None:
        valres = dict(val_loss=args.val_loss,
                      val_perplexity=_m.exp(min(args.val_loss, 20)),
                      val_tokens=args.val_tokens)
        if args.train_loss is not None:
            valres["train_loss"] = args.train_loss
            valres["train_perplexity"] = _m.exp(min(args.train_loss, 20))
            valres["train_tokens"] = args.train_tokens
    builder = br.build_markdown_zh if args.lang == "zh" else br.build_markdown
    md = builder(parsed, arch, valres, cost=cost, conv=conv, mtp_weight=mtp_w,
                 header=hdr, world_size=args.world_size, gpu_tflops=args.gpu_tflops)
    build_pdf(md, charts, args.out, args.lang, arch_fig=args.arch_fig)
    print("wrote", args.out, "with", len(charts), "charts")


if __name__ == "__main__":
    main()
