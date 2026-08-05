"""Professional local renderers for PhD application deliverables."""

from __future__ import annotations

from io import BytesIO
import re
from typing import Any
import zipfile
from xml.sax.saxutils import escape as xml_escape


ACCENT = "17324D"
TEXT = "17212B"
MUTED = "5D6B78"
LIGHT = "DDE5EC"


def render_application_pdf(
    title: str,
    content: str,
    *,
    artifact_type: str,
    profile: dict[str, Any],
    application: dict[str, Any],
) -> bytes:
    """Render a polished, selectable-text A4 PDF entirely on device."""

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        ListFlowable,
        ListItem,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title=title,
        author=_profile_text(profile, "full_name") or "OpenAgent user",
        subject=_application_label(application),
        creator="OpenAgent local document renderer",
        invariant=1,
        pageCompression=1,
    )
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ApplicationTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20 if artifact_type == "tailored_cv" else 17,
            leading=24,
            textColor=colors.HexColor(f"#{ACCENT}"),
            alignment=TA_CENTER if artifact_type == "tailored_cv" else TA_LEFT,
            spaceAfter=7,
        ),
        "contact": ParagraphStyle(
            "ApplicationContact",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=12,
            textColor=colors.HexColor(f"#{MUTED}"),
            alignment=TA_CENTER if artifact_type == "tailored_cv" else TA_LEFT,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "ApplicationHeadingOne",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=12.2,
            leading=15,
            textColor=colors.HexColor(f"#{ACCENT}"),
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ApplicationHeadingTwo",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.6,
            leading=13,
            textColor=colors.HexColor(f"#{TEXT}"),
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ApplicationBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14.2,
            textColor=colors.HexColor(f"#{TEXT}"),
            alignment=TA_LEFT,
            spaceAfter=6,
            allowWidows=0,
            allowOrphans=0,
        ),
        "bullet": ParagraphStyle(
            "ApplicationBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.7,
            leading=13.5,
            textColor=colors.HexColor(f"#{TEXT}"),
            leftIndent=3,
            firstLineIndent=0,
            spaceAfter=2,
        ),
        "small": ParagraphStyle(
            "ApplicationSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor(f"#{MUTED}"),
        ),
    }

    story: list[Any] = []
    name = _profile_text(profile, "full_name")
    contact = _contact_line(profile)
    if artifact_type == "tailored_cv" and name:
        story.append(Paragraph(_inline_markup(name), styles["title"]))
        if contact:
            story.append(Paragraph(_inline_markup(contact), styles["contact"]))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor(f"#{ACCENT}"), spaceAfter=7))
    elif artifact_type in {"cover_letter", "motivation_letter", "professor_email"}:
        if name:
            story.append(Paragraph(_inline_markup(name), styles["h1"]))
        if contact:
            story.append(Paragraph(_inline_markup(contact), styles["contact"]))
        story.append(Spacer(1, 5))

    lines = _portable_text(content).replace("\r\n", "\n").split("\n")
    if artifact_type == "tailored_cv" and lines and lines[0].lstrip().startswith("#"):
        first_heading = lines[0].lstrip("# ").strip()
        if name and first_heading.casefold() == name.casefold():
            lines = lines[1:]
    pending_bullets: list[str] = []

    def flush_bullets() -> None:
        if not pending_bullets:
            return
        story.append(
            ListFlowable(
                [ListItem(Paragraph(_inline_markup(item), styles["bullet"]), leftIndent=9) for item in pending_bullets],
                bulletType="bullet",
                start="circle",
                leftIndent=16,
                bulletFontName="Helvetica",
                bulletFontSize=6,
                bulletColor=colors.HexColor(f"#{ACCENT}"),
                spaceAfter=4,
            )
        )
        pending_bullets.clear()

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_bullets()
            story.append(Spacer(1, 3))
            continue
        if line in {"---", "***"}:
            flush_bullets()
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor(f"#{LIGHT}"), spaceBefore=3, spaceAfter=5))
            continue
        if line.startswith(('- ', '* ', '+ ')):
            pending_bullets.append(line[2:].strip())
            continue
        flush_bullets()
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            heading_text = heading.group(2).strip()
            if level == 1 and not story:
                story.append(Paragraph(_inline_markup(heading_text), styles["title"]))
            else:
                story.append(Paragraph(_inline_markup(heading_text), styles["h1" if level <= 2 else "h2"]))
            continue
        numbered = re.match(r"^\d+[.)]\s+(.+)$", line)
        if numbered:
            story.append(Paragraph(_inline_markup(line), styles["bullet"]))
            continue
        if line == "[PAGE BREAK]":
            story.append(PageBreak())
            continue
        story.append(Paragraph(_inline_markup(line), styles["body"]))
    flush_bullets()

    def page_chrome(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor(f"#{LIGHT}"))
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, 13 * mm, A4[0] - 20 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(f"#{MUTED}"))
        canvas.drawString(20 * mm, 8.5 * mm, (_application_label(application) or title)[:72])
        canvas.drawRightString(A4[0] - 20 * mm, 8.5 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story or [Paragraph(_inline_markup(title), styles["title"])], onFirstPage=page_chrome, onLaterPages=page_chrome)
    return buffer.getvalue()


def render_application_docx(
    title: str,
    content: str,
    *,
    artifact_type: str,
    profile: dict[str, Any],
    application: dict[str, Any],
) -> bytes:
    """Create a styled, editable DOCX without requiring Microsoft Word."""

    paragraphs: list[str] = []
    name = _profile_text(profile, "full_name")
    contact = _contact_line(profile)
    lines = _portable_text(content).replace("\r\n", "\n").split("\n")
    if artifact_type == "tailored_cv" and name:
        paragraphs.append(_word_paragraph(name, "Title"))
        if contact:
            paragraphs.append(_word_paragraph(contact, "Contact"))
        if lines and lines[0].lstrip().startswith("#") and lines[0].lstrip("# ").strip().casefold() == name.casefold():
            lines = lines[1:]
    elif artifact_type in {"cover_letter", "motivation_letter", "professor_email"}:
        if name:
            paragraphs.append(_word_paragraph(name, "Heading1"))
        if contact:
            paragraphs.append(_word_paragraph(contact, "Contact"))

    for raw in lines:
        line = raw.strip()
        if not line:
            paragraphs.append(_word_paragraph("", "Body"))
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            style = "Title" if level == 1 and not paragraphs else ("Heading1" if level <= 2 else "Heading2")
            paragraphs.append(_word_paragraph(heading.group(2).strip(), style))
        elif line.startswith(('- ', '* ', '+ ')):
            paragraphs.append(_word_paragraph(line[2:].strip(), "Bullet", bullet=True))
        elif re.match(r"^\d+[.)]\s+", line):
            paragraphs.append(_word_paragraph(line, "Bullet"))
        elif line not in {"---", "***"}:
            paragraphs.append(_word_paragraph(line, "Body"))

    section = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1080" w:right="1134" w:bottom="1080" w:left="1134" w:header="500" w:footer="500"/>'
        '<w:cols w:space="720"/><w:docGrid w:linePitch="360"/></w:sectPr>'
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(paragraphs)}{section}</w:body></w:document>"
    )
    styles_xml = _word_styles()
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{xml_escape(title)}</dc:title><dc:creator>{xml_escape(name or 'OpenAgent user')}</dc:creator>"
        '<cp:lastModifiedBy>OpenAgent local document renderer</cp:lastModifiedBy></cp:coreProperties>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '</Types>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '</Relationships>'
    )
    document_relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
    )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _stable_zip_write(archive, "[Content_Types].xml", content_types)
        _stable_zip_write(archive, "_rels/.rels", relationships)
        _stable_zip_write(archive, "docProps/core.xml", core_xml)
        _stable_zip_write(archive, "word/document.xml", document_xml)
        _stable_zip_write(archive, "word/styles.xml", styles_xml)
        _stable_zip_write(archive, "word/_rels/document.xml.rels", document_relationships)
    return buffer.getvalue()


def _word_paragraph(text: str, style: str, *, bullet: bool = False) -> str:
    properties = f'<w:pStyle w:val="{style}"/>'
    if bullet:
        properties += '<w:ind w:left="360" w:hanging="180"/>'
        text = f"- {text}"
    return f"<w:p><w:pPr>{properties}</w:pPr>{_word_runs(text)}</w:p>"


def _word_runs(text: str) -> str:
    parts = re.split(r"(\*\*.+?\*\*)", text)
    runs: list[str] = []
    for part in parts:
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        value = part[2:-2] if bold else part
        properties = "<w:rPr><w:b/></w:rPr>" if bold else ""
        runs.append(f'<w:r>{properties}<w:t xml:space="preserve">{xml_escape(value)}</w:t></w:r>')
    return "".join(runs) or '<w:r><w:t xml:space="preserve"></w:t></w:r>'


def _word_styles() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:color w:val="{TEXT}"/><w:sz w:val="20"/><w:lang w:val="en-US"/></w:rPr></w:rPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Body"><w:name w:val="Body"/><w:pPr><w:spacing w:after="120" w:line="284" w:lineRule="auto"/><w:widowControl/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="100"/></w:pPr><w:rPr><w:rFonts w:ascii="Aptos Display" w:hAnsi="Aptos Display"/><w:b/><w:color w:val="{ACCENT}"/><w:sz w:val="38"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Contact"><w:name w:val="Contact"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="140"/><w:pBdr><w:bottom w:val="single" w:sz="6" w:space="8" w:color="{LIGHT}"/></w:pBdr></w:pPr><w:rPr><w:color w:val="{MUTED}"/><w:sz w:val="17"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:basedOn w:val="Body"/><w:pPr><w:keepNext/><w:spacing w:before="180" w:after="80"/><w:pBdr><w:bottom w:val="single" w:sz="4" w:space="4" w:color="{LIGHT}"/></w:pBdr></w:pPr><w:rPr><w:b/><w:color w:val="{ACCENT}"/><w:sz w:val="25"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/><w:basedOn w:val="Body"/><w:pPr><w:keepNext/><w:spacing w:before="130" w:after="50"/></w:pPr><w:rPr><w:b/><w:color w:val="{TEXT}"/><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Bullet"><w:name w:val="Bullet"/><w:basedOn w:val="Body"/><w:pPr><w:spacing w:after="45" w:line="270" w:lineRule="auto"/><w:ind w:left="360" w:hanging="180"/></w:pPr></w:style>
</w:styles>'''


def _stable_zip_write(archive: zipfile.ZipFile, name: str, value: str) -> None:
    info = zipfile.ZipInfo(name, date_time=(2024, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, value.encode("utf-8"))


def _inline_markup(value: str) -> str:
    escaped = xml_escape(value)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*", r"<i>\1</i>", escaped)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    return escaped


def _portable_text(value: str) -> str:
    return (
        value.replace("\u2014", " - ")
        .replace("\u2013", " - ")
        .replace("\u2011", "-")
        .replace("\u00a0", " ")
    )


def _profile_text(profile: dict[str, Any], key: str) -> str:
    value = profile.get(key, "")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip() if value else ""


def _contact_line(profile: dict[str, Any]) -> str:
    return " | ".join(
        item
        for item in (
            _profile_text(profile, "email"),
            _profile_text(profile, "phone"),
            _profile_text(profile, "location"),
            _profile_text(profile, "links"),
        )
        if item
    )


def _application_label(application: dict[str, Any]) -> str:
    position = str(application.get("position_title") or application.get("program") or "PhD application").strip()
    institution = str(application.get("institution") or "").strip()
    return f"{position} at {institution}" if institution else position
