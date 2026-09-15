"""StudyAgent AI domain rules shared by API, RAG, and UI."""

from .curriculum import (
    SEMESTER_LABELS,
    SUBJECTS,
    allowed_subjects,
    grade_label,
    semester_label,
    textbook_filename,
    textbook_id,
)

__all__ = [
    "SEMESTER_LABELS",
    "SUBJECTS",
    "allowed_subjects",
    "grade_label",
    "semester_label",
    "textbook_filename",
    "textbook_id",
]
