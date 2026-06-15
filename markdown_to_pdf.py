import re
from markdown_pdf import MarkdownPdf, Section
from pathlib import Path

# this module normalize the markdown search results
# and convert it to a format proper for pdf file

def normalize_markdown_headings(markdown_content: str) -> str:
    """PyMuPDF TOC requires h1 first and no skipped heading levels (e.g. # then ####)."""
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    lines = markdown_content.splitlines()
    headings = []
    for i, line in enumerate(lines):
        match = heading_re.match(line)
        if match:
            headings.append((i, len(match.group(1)), match.group(2)))

    if not headings:
        return "# Research Report\n\n" + markdown_content

    prev_level = 0
    for idx, (line_i, level, title) in enumerate(headings):
        if idx == 0:
            new_level = 1
        else:
            new_level = min(max(level, 1), prev_level + 1)
        lines[line_i] = "#" * new_level + " " + title
        prev_level = new_level

    return "\n".join(lines)

def md_to_pdf(markdown_content: str):
    """Useful when you want to generate a pdf file from markdown content"""
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / "report.pdf"
    normalized = normalize_markdown_headings(markdown_content)
    pdf = MarkdownPdf()
    pdf.add_section(Section(normalized))
    pdf.meta["title"] = "Generated Report"
    pdf.save(str(pdf_path))
    return str(pdf_path)