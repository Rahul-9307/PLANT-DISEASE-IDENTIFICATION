from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import HRFlowable
import io
from datetime import datetime

def generate_pdf(disease, confidence, info):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontName="STSong-Light",
        fontSize=20,
        textColor=colors.white,
        alignment=1,
        spaceAfter=15
    )

    normal_style = ParagraphStyle(
        name='NormalStyle',
        parent=styles['Normal'],
        fontName="STSong-Light",
        fontSize=12,
        spaceAfter=8
    )

    # ---------------- Header Background Box ----------------
    header_data = [[Paragraph("🌾 AGRISENS - PLANT DISEASE REPORT", title_style)]]

    header_table = Table(header_data, colWidths=[450])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.green),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0, colors.white),
        ('BOX', (0, 0), (-1, -1), 0, colors.white),
    ]))

    elements.append(header_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Date
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d-%m-%Y %H:%M')}", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Disease Summary Table
    summary_data = [
        ["Disease", disease],
        ["Confidence", f"{confidence:.2f}%"]
    ]

    summary_table = Table(summary_data, colWidths=[150, 300])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), "STSong-Light"),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Severity Indicator
    if confidence > 85:
        severity = "HIGH SEVERITY INFECTION"
        severity_color = colors.red
    elif confidence > 60:
        severity = "MODERATE INFECTION"
        severity_color = colors.orange
    else:
        severity = "LOW INFECTION LEVEL"
        severity_color = colors.green

    severity_style = ParagraphStyle(
        name='SeverityStyle',
        parent=styles['Heading2'],
        fontName="STSong-Light",
        fontSize=14,
        textColor=severity_color,
        alignment=1,
        spaceAfter=20
    )

    elements.append(Paragraph(severity, severity_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 0.3 * inch))

    # Detailed Sections
    sections = [
        ("Description", info["description"]),
        ("Cause", info["cause"]),
        ("Prevention", info["prevention"]),
        ("Treatment", info["treatment"]),
    ]

    for title, content in sections:
        elements.append(Paragraph(f"<b>{title}</b>", normal_style))
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(content, normal_style))
        elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer
