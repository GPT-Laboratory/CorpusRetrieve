"""Synthetic documents for the regression suite.

``bilingual_report`` reproduces, on four pages, every failure class the
industrial evaluation identified: Finnish IMRaD headings, a ruled table captioned
"Taulukko 1", a logo repeated on every page, a figure captioned "Kuva 1",
an English results section, percentages in every common form, and a
suspended-hyphen compound ("maa- ja metsätalous").
"""
from __future__ import annotations

import io
from pathlib import Path

FONT = "helv"


def _png(color: tuple[int, int, int], size=(120, 90)) -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, color)
    d = ImageDraw.Draw(img)
    d.line((0, 0, size[0], size[1]), fill=(0, 0, 0), width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _text(page, x, y, text, size=10):
    page.insert_text((x, y), text, fontsize=size, fontname=FONT)


def bilingual_report(path: Path) -> Path:
    import fitz
    logo = _png((200, 30, 30), (80, 60))
    chart = _png((30, 90, 200), (300, 200))
    doc = fitz.open()

    # page 1: Finnish introduction with a suspended hyphen
    p = doc.new_page()
    p.insert_image(fitz.Rect(480, 20, 560, 80), stream=logo)
    _text(p, 72, 110, "1 Johdanto", 16)
    _text(p, 72, 140, "Laite- ja ohjelmistotestaus kasvoi 35 % vuonna 2015.")
    _text(p, 72, 155, "Ääriarvot olivat harvinaisia. Epäonnistumisten osuus laski 3,5 %.")

    # page 2: Finnish methods + a ruled table captioned "Taulukko 1"
    p = doc.new_page()
    p.insert_image(fitz.Rect(480, 20, 560, 80), stream=logo)
    _text(p, 72, 110, "2 Aineisto ja menetelmät", 16)
    _text(p, 72, 140, "Kokeet tehtiin kolmessa paikassa vuosina 2012-2014.")
    _text(p, 72, 180, "Taulukko 1. Mittaukset toimipaikoittain.")
    x0, y0, cw, rh = 72, 190, 140, 20
    rows = [["Paikka", "Mittaus kWh", "Virta A"], ["A", "4200", "90"],
            ["B", "3900", "80"], ["C", "4500", "100"]]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            p.draw_rect(rect, color=(0, 0, 0), width=0.8)
            _text(p, rect.x0 + 4, rect.y1 - 6, cell, 9)
    _text(p, 72, 300, "Taulukossa 1 esitetään mittaukset.")

    # page 3: English results with a figure captioned "Kuva 1"
    p = doc.new_page()
    p.insert_image(fitz.Rect(480, 20, 560, 80), stream=logo)
    _text(p, 72, 110, "3 Results", 16)
    _text(p, 72, 140, "Performance rose 35% compared to 2010 (Smith et al., 2012).")
    _text(p, 72, 155, "The share increased from 60% to 85% in the trial.")
    p.insert_image(fitz.Rect(72, 180, 372, 380), stream=chart)
    _text(p, 72, 395, "Kuva 1. Mittauksen kehitys.")

    # page 4: references
    p = doc.new_page()
    p.insert_image(fitz.Rect(480, 20, 560, 80), stream=logo)
    _text(p, 72, 110, "Lähteet", 16)
    _text(p, 72, 140, "Smith, J. 2012. Instrument accuracy in laboratories. Journal of Measurement 4: 1-10.")
    doc.save(str(path))
    doc.close()
    return path


def blank_pdf(path: Path, pages: int = 1) -> Path:
    import fitz
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(str(path))
    doc.close()
    return path
