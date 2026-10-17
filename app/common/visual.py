"""Side-by-side comparison image: scribbled input, result, prompt caption."""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def make_visual(
    left: Image.Image, right: Image.Image, caption: str, *, original: Image.Image | None = None
) -> Image.Image:
    """Show overlay/result, optionally preceded by the clean original."""
    panels = [("Input + scribble", left), ("Output", right)]
    if original is not None:
        panels.insert(0, ("Original", original))
    target_h = right.height
    resized = []
    for label, panel in panels:
        panel = panel.convert("RGB")
        if panel.height != target_h:
            panel = panel.resize(
                (max(1, round(panel.width * target_h / panel.height)), target_h),
                Image.Resampling.LANCZOS,
            )
        resized.append((label, panel))

    gap = 16
    margin = 16
    label_h = 36
    label_font = _font(22)
    caption_font = _font(20)
    canvas_w = margin * 2 + sum(panel.width for _, panel in resized) + gap * (len(resized) - 1)
    text_width = max(1, canvas_w - margin * 2)
    lines = _wrap(caption.strip(), caption_font, text_width) or [""]
    line_h = _line_height(caption_font)
    caption_h = line_h * len(lines)

    canvas_h = margin + label_h + target_h + 12 + caption_h + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y_img = margin + label_h
    x = margin
    for label, panel in resized:
        draw.text((x, margin), label, fill=(0, 0, 0), font=label_font)
        canvas.paste(panel, (x, y_img))
        x += panel.width + gap

    y = y_img + target_h + 12
    for line in lines:
        draw.text((margin, y), line, fill=(20, 20, 20), font=caption_font)
        y += line_h
    return canvas


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _line_height(font: ImageFont.ImageFont) -> int:
    try:
        box = font.getbbox("Ag")
        return max(1, box[3] - box[1]) + 8
    except Exception:
        return 28


def _text_width(text: str, font: ImageFont.ImageFont) -> int:
    if hasattr(font, "getlength"):
        return int(font.getlength(text))
    box = font.getbbox(text)
    return box[2] - box[0]


def _wrap(text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if _text_width(trial, font) <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    if lines and _text_width(lines[0], font) > width:
        # Bitmap default fonts have no reliable width; fall back to characters.
        lines = textwrap.wrap(text, width=100) or [text]
    return lines
