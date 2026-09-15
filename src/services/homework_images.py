"""Create private, user-scoped question crops from uploaded homework pages."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageOps

from src.agents.homework.models import ExtractedQuestion, GradedQuestion


def _major_key(question: ExtractedQuestion | GradedQuestion) -> str:
    major = str(question.major_number or "").strip()
    if major:
        return f"major-{major}"
    return f"question-{question.page_index}-{question.number}"


def _safe_box(value: list[int]) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = [max(0, min(1000, int(part))) for part in value]
    except (TypeError, ValueError):
        return None
    if right - left < 10 or bottom - top < 10:
        return None
    return left, top, right, bottom


def create_question_crops(
    questions: list[ExtractedQuestion] | list[GradedQuestion],
    source_paths: list[str],
    batch_id: str,
) -> dict[str, list[Path]]:
    """Crop every printed major question, including all of its subquestions.

    Bounding boxes use a 0..1000 coordinate system.  All boxes belonging to the
    same major question on one page are unioned, so a crop never separates a
    stem, diagram, or sibling subquestion.  If the model cannot provide a
    reliable box, the complete uploaded page is retained as the safe fallback.
    """
    by_group_page: dict[tuple[str, int], list[tuple[int, int, int, int] | None]] = {}
    group_pages: dict[str, set[int]] = {}
    for question in questions:
        page = int(question.page_index)
        if page < 0 or page >= len(source_paths):
            continue
        key = _major_key(question)
        by_group_page.setdefault((key, page), []).append(_safe_box(question.bounding_box))
        group_pages.setdefault(key, set()).add(page)

    crops: dict[str, list[Path]] = {}
    for key, pages in group_pages.items():
        for page in sorted(pages):
            source = Path(source_paths[page])
            if not source.is_file():
                continue
            boxes = by_group_page[(key, page)]
            safe_key = re.sub(r"[^A-Za-z0-9_-]+", "-", key)[:60]
            target = source.parent / f"crop_{batch_id}_{safe_key}_p{page + 1}.jpg"
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                width, height = image.size
                valid = [box for box in boxes if box is not None]
                # Any missing child box means a tight crop could silently omit
                # part of the printed major question. Keep the complete page.
                if valid and len(valid) == len(boxes):
                    left = min(box[0] for box in valid)
                    top = min(box[1] for box in valid)
                    right = max(box[2] for box in valid)
                    bottom = max(box[3] for box in valid)
                    pad_x, pad_y = 24, 24
                    crop_box = (
                        max(0, round(width * (left - pad_x) / 1000)),
                        max(0, round(height * (top - pad_y) / 1000)),
                        min(width, round(width * (right + pad_x) / 1000)),
                        min(height, round(height * (bottom + pad_y) / 1000)),
                    )
                    image = image.crop(crop_box)
                    image.save(target, format="JPEG", quality=92, optimize=True)
                    crop_path = target
                else:
                    # Reuse the already-saved page instead of creating one full
                    # page duplicate per major question.
                    crop_path = source
            crops.setdefault(key, []).append(crop_path)
    return crops


def attach_question_crops(
    questions: list[GradedQuestion],
    crops: dict[str, list[Path]],
) -> None:
    """Attach public URLs plus private paths to graded questions in place."""
    for question in questions:
        paths = crops.get(_major_key(question), [])
        question.question_image_paths = [str(path) for path in paths]
        question.question_image_urls = [
            f"/api/v1/homework/images/{path.name}" for path in paths
        ]


__all__ = ["attach_question_crops", "create_question_crops"]
