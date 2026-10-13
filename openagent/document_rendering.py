"""Render plain chat documents as PDF or editable DOCX files."""

from __future__ import annotations

from io import BytesIO
import re
import textwrap
import zipfile
from xml.sax.saxutils import escape


def render_document_pdf(title: str, content: str) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import HRFlowable, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer
    except ModuleNotFoundError as exc:
        if exc.name != "reportlab":
            raise
        return _basic_pdf(title, content)

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=19 * mm, bottomMargin=18 * mm, title=title,
        author="OpenAgent user", subject=title, creator="OpenAgent local document renderer",
        invariant=1, pageCompression=1,
    )
    base = getSampleStyleSheet()
    heading = ParagraphStyle("DocumentHeading", parent=base["Heading1"], textColor=colors.HexColor("#17324D"), keepWithNext=True)
    body = ParagraphStyle("DocumentBody", parent=base["BodyText"], fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#17212B"), spaceAfter=6)
    story = [Paragraph(_markup(title), ParagraphStyle("DocumentTitle", parent=base["Title"], textColor=colors.HexColor("#17324D"), alignment=0, spaceAfter=12))]
    bullets: list[str] = []

    def flush_bullets() -> None:
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(_markup(line), body)) for line in bullets], bulletType="bullet", leftIndent=16, spaceAfter=5))
            bullets.clear()

    for raw in content.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            flush_bullets()
            story.append(Spacer(1, 3))
        elif line in {"---", "***"}:
            flush_bullets()
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#DDE5EC"), spaceBefore=3, spaceAfter=5))
        elif line.startswith(("- ", "* ", "+ ")):
            bullets.append(line[2:].strip())
        else:
            flush_bullets()
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                level = len(match.group(1))
                story.append(Paragraph(_markup(match.group(2)), heading if level < 3 else body))
            elif re.match(r"^\d+[.)]\s+", line):
                story.append(Paragraph(_markup(line), body))
            else:
                story.append(Paragraph(_markup(line), body))
    flush_bullets()

    def page_chrome(canvas, page) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#DDE5EC"))
        canvas.line(20 * mm, 13 * mm, A4[0] - 20 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#5D6B78"))
        canvas.drawString(20 * mm, 8.5 * mm, title[:72])
        canvas.drawRightString(A4[0] - 20 * mm, 8.5 * mm, f"Page {page.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    return buffer.getvalue()


def render_document_docx(title: str, content: str) -> bytes:
    paragraphs = [_paragraph(title, "Title")]
    for raw in content.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            paragraphs.append(_paragraph("", "Body"))
        elif line in {"---", "***"}:
            continue
        elif match := re.match(r"^(#{1,6})\s+(.+)$", line):
            paragraphs.append(_paragraph(match.group(2), "Heading1" if len(match.group(1)) <= 2 else "Heading2"))
        elif line.startswith(("- ", "* ", "+ ")):
            paragraphs.append(_paragraph(line[2:], "Bullet", bullet=True))
        else:
            paragraphs.append(_paragraph(line, "Body"))
    section = '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1080" w:right="1134" w:bottom="1080" w:left="1134"/></w:sectPr>'
    document_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + "".join(paragraphs) + section + "</w:body></w:document>"
    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="20"/></w:rPr></w:rPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Body"><w:name w:val="Body"/><w:pPr><w:spacing w:after="120" w:line="284" w:lineRule="auto"/></w:pPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:spacing w:after="180"/></w:pPr><w:rPr><w:b/><w:color w:val="17324D"/><w:sz w:val="36"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:basedOn w:val="Body"/><w:pPr><w:keepNext/><w:spacing w:before="180" w:after="80"/></w:pPr><w:rPr><w:b/><w:color w:val="17324D"/><w:sz w:val="25"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/><w:basedOn w:val="Body"/><w:pPr><w:keepNext/><w:spacing w:before="130" w:after="50"/></w:pPr><w:rPr><w:b/><w:sz w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Bullet"><w:name w:val="Bullet"/><w:basedOn w:val="Body"/><w:pPr><w:ind w:left="360" w:hanging="180"/></w:pPr></w:style></w:styles>'''
    content_types = '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
    relationships = '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    document_relationships = '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _zip_write(archive, "[Content_Types].xml", content_types)
        _zip_write(archive, "_rels/.rels", relationships)
        _zip_write(archive, "word/document.xml", document_xml)
        _zip_write(archive, "word/styles.xml", styles_xml)
        _zip_write(archive, "word/_rels/document.xml.rels", document_relationships)
    return buffer.getvalue()


def _paragraph(text: str, style: str, *, bullet: bool = False) -> str:
    props = f'<w:pStyle w:val="{style}"/>'
    if bullet:
        props += '<w:ind w:left="360" w:hanging="180"/>'
        text = f"- {text}"
    parts = re.split(r"(\*\*.+?\*\*)", text)
    runs = []
    for part in parts:
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        value = part[2:-2] if bold else part
        formatting = "<w:rPr><w:b/></w:rPr>" if bold else ""
        runs.append(f'<w:r>{formatting}<w:t xml:space="preserve">{escape(value)}</w:t></w:r>')
    return f"<w:p><w:pPr>{props}</w:pPr>{''.join(runs)}</w:p>"


def _zip_write(archive: zipfile.ZipFile, name: str, text: str) -> None:
    info = zipfile.ZipInfo(name, date_time=(2024, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, text.encode("utf-8"))


def _markup(value: str) -> str:
    value = escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*", r"<i>\1</i>", value)
    return re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)


def _basic_pdf(title: str, content: str) -> bytes:
    text = (title + "\n\n" + content).encode("latin-1", errors="replace").decode("latin-1")
    lines = [line for raw in text.splitlines() for line in (textwrap.wrap(raw, 92) or [""])]
    page_lines = [lines[index:index + 55] for index in range(0, max(1, len(lines)), 55)]
    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + index * 2} 0 R" for index in range(len(page_lines)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_lines)} >>")
    font_id = 3 + len(page_lines) * 2
    for index, lines_on_page in enumerate(page_lines):
        page_id = 3 + index * 2
        content_id = page_id + 1
        commands = ["BT /F1 10 Tf 42 790 Td"]
        commands.extend(f"({_pdf_escape(line)}) Tj 0 -13 Td" for line in lines_on_page)
        commands.append("ET")
        stream = " ".join(commands)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>")
        objects.append(f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, item in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n{item}\nendobj\n".encode("latin-1"))
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode("ascii"))
    return bytes(output)


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
