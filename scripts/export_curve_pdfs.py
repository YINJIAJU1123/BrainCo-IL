#!/usr/bin/env python3
"""Package exported curve PNGs into clear, shareable per-episode PDFs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re

from PIL import Image
from reportlab.lib.pagesizes import A3
from reportlab.lib.pagesizes import landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

EPISODE_PATTERN = re.compile(r"^ep(?P<episode>\d+)_dim(?P<dimension>\d+)_(?P<joint>.+)\.png$")


def _title_page(pdf: canvas.Canvas, title: str, subtitle: str, page_size: tuple[float, float]) -> None:
    width, height = page_size
    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawCentredString(width / 2, height * 0.68, title)
    pdf.setFont("Helvetica", 18)
    pdf.drawCentredString(width / 2, height * 0.59, subtitle)
    pdf.setFont("Helvetica", 11)
    lines = [
        "Blue: observation.state (measured)    Orange: action (target)",
        "Panels: position, angle step, velocity, acceleration",
        "Thresholds: 0.35 rad/frame, 6 rad/s, 100 rad/s^2",
        "Source: original LeRobot v2.1 Parquet values; no smoothing or normalization",
    ]
    for index, line in enumerate(lines):
        pdf.drawCentredString(width / 2, height * 0.46 - index * 22, line)
    pdf.showPage()


def _curve_page(
    pdf: canvas.Canvas,
    image_path: Path,
    page_size: tuple[float, float],
    page_number: int,
    total_pages: int,
) -> None:
    width, height = page_size
    margin = 22
    footer = 16
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    available_width = width - 2 * margin
    available_height = height - 2 * margin - footer
    scale = min(available_width / image_width, available_height / image_height)
    draw_width = image_width * scale
    draw_height = image_height * scale
    x = (width - draw_width) / 2
    y = margin + footer + (available_height - draw_height) / 2
    pdf.drawImage(ImageReader(str(image_path)), x, y, draw_width, draw_height, preserveAspectRatio=True)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(margin, 10, image_path.stem)
    pdf.drawRightString(width - margin, 10, f"{page_number}/{total_pages}")
    pdf.showPage()


def _write_pdf(output: Path, images: list[Path], title: str, subtitle: str) -> None:
    page_size = landscape(A3)
    pdf = canvas.Canvas(str(output), pagesize=page_size, pageCompression=1)
    pdf.setTitle(title)
    _title_page(pdf, title, subtitle, page_size)
    for page_number, image_path in enumerate(images, start=1):
        _curve_page(pdf, image_path, page_size, page_number, len(images))
    pdf.save()


def export_pdfs(curves_dir: Path, output_dir: Path) -> None:
    grouped: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted((curves_dir / "plots").glob("*.png")):
        match = EPISODE_PATTERN.match(path.name)
        if match:
            grouped[int(match.group("episode"))].append((int(match.group("dimension")), path))
    if not grouped:
        raise ValueError(f"No curve PNGs found under {curves_dir / 'plots'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_images: list[Path] = []
    for episode, entries in sorted(grouped.items()):
        images = [path for _, path in sorted(entries)]
        all_images.extend(images)
        _write_pdf(
            output_dir / f"revo3_arm_curves_episode_{episode:02d}.pdf",
            images,
            "Revo3 Arm Joint Curves",
            f"Episode {episode} - {len(images)} arm joints - full episode",
        )
    _write_pdf(
        output_dir / "revo3_arm_curves_all_episodes.pdf",
        all_images,
        "Revo3 Arm Joint Curves",
        f"All episodes - {len(all_images)} joint curves",
    )
    print(f"Exported {len(grouped)} per-episode PDFs and one combined PDF to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curves_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_pdfs(args.curves_dir.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
