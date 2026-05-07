from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import os
import re

INPUT = os.path.join("docs", "documentation.md")
OUTPUT = os.path.join("docs", "documentation.pdf")

def md_inline_to_html(text):
    # basic replacements for bold and italic
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    return text


def parse_markdown_to_flowables(md_text):
    styles = getSampleStyleSheet()
    normal = styles['BodyText']
    heading1 = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=18, leading=22)
    heading2 = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=14, leading=18)
    flowables = []

    lines = md_text.splitlines()
    buffer = []
    in_list = False
    list_items = []

    def flush_paragraph():
        nonlocal buffer
        if buffer:
            text = ' '.join(buffer).strip()
            if text:
                flowables.append(Paragraph(md_inline_to_html(text), normal))
                flowables.append(Spacer(1, 6))
            buffer = []

    for line in lines:
        s = line.strip()
        if not s:
            # blank line: flush paragraph or list
            if in_list:
                if list_items:
                    flowables.append(ListFlowable([
                        ListItem(Paragraph(md_inline_to_html(it), normal), leftIndent=12) for it in list_items
                    ], bulletType='bullet'))
                    flowables.append(Spacer(1,6))
                list_items = []
                in_list = False
            else:
                flush_paragraph()
            continue

        # headings
        if s.startswith('#'):
            flush_paragraph()
            hashes = len(s) - len(s.lstrip('#'))
            heading_text = s.lstrip('#').strip()
            if hashes == 1:
                flowables.append(Paragraph(md_inline_to_html(heading_text), heading1))
            else:
                flowables.append(Paragraph(md_inline_to_html(heading_text), heading2))
            flowables.append(Spacer(1, 8))
            continue

        # list item
        if s.startswith('- ') or s.startswith('* '):
            in_list = True
            list_items.append(s[2:].strip())
            continue

        # horizontal rule
        if s.startswith('---'):
            flush_paragraph()
            flowables.append(Spacer(1, 6))
            continue

        # normal text
        buffer.append(s)

    # flush any remaining
    if in_list and list_items:
        flowables.append(ListFlowable([
            ListItem(Paragraph(md_inline_to_html(it), normal), leftIndent=12) for it in list_items
        ], bulletType='bullet'))
    else:
        flush_paragraph()

    return flowables


def main():
    if not os.path.exists(INPUT):
        print(f"Input file not found: {INPUT}")
        return

    with open(INPUT, 'r', encoding='utf-8') as f:
        md = f.read()

    doc = SimpleDocTemplate(OUTPUT, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    flowables = parse_markdown_to_flowables(md)
    doc.build(flowables)
    print(f"PDF generated: {OUTPUT}")

if __name__ == '__main__':
    main()
