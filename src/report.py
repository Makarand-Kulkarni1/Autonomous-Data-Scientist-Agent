from pathlib import Path

from src.orchestrator import RunResult


def generate_report(result: RunResult, goal: str, workspace: Path, output_path: Path) -> Path:
    """Turn a RunResult into one readable Markdown file."""
    lines = [
        f"# Auto Data Scientist Report",
        f"",
        f"**Goal:** {goal}",
        f"**Problem type:** {result.plan.problem_type}",
        f"**Target column:** {result.plan.target_column}",
        f"**Overall result:** {'SUCCESS' if result.success else f'FAILED at step {result.failed_at_step}'}",
        f"",
        f"## Steps",
    ]

    for log in result.logs:
        status = "PASSED" if log.approved else "FAILED"
        lines += [
            f"",
            f"### Step {log.step_id}: {log.title} — {status}",
            f"",
            f"**Task:** {log.task}",
            f"",
            f"**Coder attempts:** {log.coder_attempts}",
            f"**Reviewer verdict:** {log.review_reason}",
        ]
        if log.leakage_flags:
            lines.append(f"**Leakage flags:** {', '.join(log.leakage_flags)}")

        if len(log.review_history) > 1:
            lines += ["", "**Review history:**", ""]
            for entry in log.review_history:
                verdict = "APPROVED" if entry["approved"] else "REJECTED"
                lines.append(f"- Attempt {entry['attempt']}: {verdict} — {entry['reason']}")

        if log.stdout.strip():
            lines += ["", "**Output:**", "```", log.stdout.strip(), "```"]

    charts = sorted(Path(workspace).glob("*.png"))
    if charts:
        lines += ["", "## Charts", ""]
        for chart in charts:
            lines.append(f"![{chart.stem}]({chart.name})")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def generate_pdf_report(result: RunResult, goal: str, workspace: Path, output_path: Path) -> Path:
    """Turn a RunResult into a PDF report, mirroring generate_report's structure."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Preformatted
    )

    output_path = Path(output_path)
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                            topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    code_style = ParagraphStyle(
        "Code", parent=styles["Normal"], fontName="Courier", fontSize=8, leading=10,
    )

    story = []

    story.append(Paragraph("Auto Data Scientist Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Goal:</b> {goal}", styles["Normal"]))
    story.append(Paragraph(f"<b>Problem type:</b> {result.plan.problem_type}", styles["Normal"]))
    story.append(Paragraph(f"<b>Target column:</b> {result.plan.target_column}", styles["Normal"]))
    overall = "SUCCESS" if result.success else f"FAILED at step {result.failed_at_step}"
    story.append(Paragraph(f"<b>Overall result:</b> {overall}", styles["Normal"]))
    story.append(Spacer(1, 20))

    for log in result.logs:
        status = "PASSED" if log.approved else "FAILED"
        story.append(Paragraph(f"Step {log.step_id}: {log.title} — {status}", styles["Heading2"]))
        story.append(Paragraph(f"<b>Task:</b> {log.task}", styles["Normal"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Coder attempts:</b> {log.coder_attempts}", styles["Normal"]))
        story.append(Paragraph(f"<b>Reviewer verdict:</b> {log.review_reason}", styles["Normal"]))

        if log.leakage_flags:
            story.append(Paragraph(f"<b>Leakage flags:</b> {', '.join(log.leakage_flags)}", styles["Normal"]))

        if len(log.review_history) > 1:
            story.append(Spacer(1, 4))
            story.append(Paragraph("<b>Review history:</b>", styles["Normal"]))
            for entry in log.review_history:
                verdict = "APPROVED" if entry["approved"] else "REJECTED"
                story.append(Paragraph(f"&bull; Attempt {entry['attempt']}: {verdict} — {entry['reason']}",
                                       styles["Normal"]))

        if log.stdout.strip():
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Output:</b>", styles["Normal"]))
            truncated = log.stdout.strip()[:2000]
            story.append(Preformatted(truncated, code_style))

        story.append(Spacer(1, 16))

    charts = sorted(Path(workspace).glob("*.png"))
    if charts:
        story.append(PageBreak())
        story.append(Paragraph("Charts", styles["Title"]))
        story.append(Spacer(1, 12))
        for chart in charts:
            story.append(Paragraph(chart.stem, styles["Heading3"]))
            try:
                img = RLImage(str(chart), width=5.5 * inch, height=3.5 * inch, kind="proportional")
                story.append(img)
            except Exception:
                story.append(Paragraph(f"(could not embed {chart.name})", styles["Normal"]))
            story.append(Spacer(1, 16))

    doc.build(story)
    return output_path
