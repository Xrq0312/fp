from __future__ import annotations

import html
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def run(text: str, bold: bool = False, size: int = 22, font: str = "Microsoft YaHei") -> str:
    bold_xml = "<w:b/>" if bold else ""
    return (
        "<w:r><w:rPr>"
        f"{bold_xml}<w:rFonts w:ascii=\"{font}\" w:eastAsia=\"Microsoft YaHei\" w:hAnsi=\"{font}\"/>"
        f"<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>"
        "</w:rPr>"
        f"<w:t xml:space=\"preserve\">{esc(text)}</w:t></w:r>"
    )


def paragraph(
    text: str = "",
    style: str = "BodyText",
    bold: bool = False,
    size: int = 22,
    align: str | None = None,
    before: int = 0,
    after: int = 100,
    font: str = "Microsoft YaHei",
) -> str:
    props = [f"<w:pStyle w:val=\"{style}\"/>"] if style else []
    props.append(f"<w:spacing w:before=\"{before}\" w:after=\"{after}\"/><w:line w:val=\"300\" w:rule=\"auto\"/>")
    if align:
        props.append(f"<w:jc w:val=\"{align}\"/>")
    return f"<w:p><w:pPr>{''.join(props)}</w:pPr>{run(text, bold, size, font)}</w:p>"


def table(rows: list[list[str]]) -> str:
    if len(rows[0]) == 3:
        widths = [2100, 3300, 4300]
    elif len(rows[0]) == 4:
        widths = [2200, 3900, 1400, 1400]
    else:
        widths = [2400, 6800]
    xml = [
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="6"/><w:left w:val="single" w:sz="6"/>'
        '<w:bottom w:val="single" w:sz="6"/><w:right w:val="single" w:sz="6"/>'
        '<w:insideH w:val="single" w:sz="6"/><w:insideV w:val="single" w:sz="6"/></w:tblBorders>'
        "</w:tblPr><w:tblGrid>"
    ]
    for width in widths:
        xml.append(f'<w:gridCol w:w="{width}"/>')
    xml.append("</w:tblGrid>")
    for row_index, row in enumerate(rows):
        xml.append("<w:tr>")
        for cell_index, cell in enumerate(row):
            shade = '<w:shd w:fill="EDEDED"/>' if row_index == 0 else ""
            width = widths[min(cell_index, len(widths) - 1)]
            xml.append(f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}</w:tcPr>')
            xml.append(paragraph(cell, bold=row_index == 0, size=19, before=0, after=0))
            xml.append("</w:tc>")
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def clean_inline(text: str) -> str:
    return text.replace("`", "")


def markdown_to_body(markdown: str) -> str:
    lines = markdown.splitlines()
    parts: list[str] = []
    i = 0
    first_title = True
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            parts.append(paragraph("\n".join(code_lines), style="CodeText", size=19, font="Consolas", before=80, after=120))
            continue
        if line.startswith("|"):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].rstrip().startswith("|"):
                cells = [clean_inline(cell.strip()) for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append(cells)
                i += 1
            if rows:
                parts.append(table(rows))
            continue
        if line.startswith("# "):
            style = "Title" if first_title else "Heading1"
            first_title = False
            parts.append(
                paragraph(
                    clean_inline(line[2:].strip()),
                    style=style,
                    bold=True,
                    size=38 if style == "Title" else 30,
                    align="center" if style == "Title" else None,
                    before=0 if style == "Title" else 240,
                    after=180 if style == "Title" else 120,
                )
            )
        elif line.startswith("## "):
            parts.append(paragraph(clean_inline(line[3:].strip()), style="Heading1", bold=True, size=30, before=240, after=120))
        elif line.startswith("### "):
            parts.append(paragraph(clean_inline(line[4:].strip()), style="Heading2", bold=True, size=25, before=180, after=80))
        elif line.startswith("- "):
            parts.append(paragraph("- " + clean_inline(line[2:].strip()), style="BodyText", size=21, after=70))
        else:
            parts.append(paragraph(clean_inline(line.strip()), style="BodyText", size=21))
        i += 1
    return "".join(parts)


def write_docx(source: Path, output: Path) -> None:
    body_xml = markdown_to_body(source.read_text(encoding="utf-8-sig"))
    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1200" w:right="1200" w:bottom="1200" w:left="1200" '
        'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
    )
    document_xml = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{W_NS}"><w:body>{body_xml}{sect}</w:body></w:document>'
    styles_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="{W_NS}">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:b/><w:sz w:val="38"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="25"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="BodyText"><w:name w:val="Body Text"/><w:basedOn w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="CodeText"><w:name w:val="Code Text"/><w:basedOn w:val="Normal"/><w:pPr><w:shd w:fill="F2F2F2"/></w:pPr><w:rPr><w:rFonts w:ascii="Consolas" w:eastAsia="Microsoft YaHei" w:hAnsi="Consolas"/><w:sz w:val="19"/></w:rPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr></w:style>
</w:styles>"""
    content_types = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>'
    rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>'
    word_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>Point Cloud Corner Technical Report</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>'
    app = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Codex</Application></Properties>'
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document_xml.encode("utf-8"))
        z.writestr("word/styles.xml", styles_xml.encode("utf-8"))
        z.writestr("word/_rels/document.xml.rels", word_rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/point_cloud_corner_technical_report_source.md")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs/point_cloud_corner_technical_report.docx")
    write_docx(src, out)
    print(out.resolve())
