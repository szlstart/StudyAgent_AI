"""The single source of truth for the grade 1-9 curriculum."""

from __future__ import annotations

from pathlib import Path

SUBJECTS: dict[str, dict[str, str]] = {
    "chinese": {"label": "语文", "icon": "📖"},
    "math": {"label": "数学", "icon": "📐"},
    "english": {"label": "英语", "icon": "🌍"},
    "physics": {"label": "物理", "icon": "⚡"},
    "biology": {"label": "生物", "icon": "🌿"},
    "history": {"label": "历史", "icon": "📜"},
    "chemistry": {"label": "化学", "icon": "🧪"},
    "geography": {"label": "地理", "icon": "🌏"},
}

SEMESTER_LABELS = {"upper": "上册", "lower": "下册"}

_GRADE_CN = {
    1: "一年级", 2: "二年级", 3: "三年级", 4: "四年级", 5: "五年级",
    6: "六年级", 7: "七年级", 8: "八年级", 9: "九年级",
}

_CURRICULUM: dict[int, tuple[str, ...]] = {
    1: ("chinese", "math"),
    2: ("chinese", "math"),
    3: ("chinese", "math", "english"),
    4: ("chinese", "math", "english"),
    5: ("chinese", "math", "english"),
    6: ("chinese", "math", "english"),
    7: ("chinese", "math", "english", "biology", "history", "geography"),
    8: ("chinese", "math", "english", "physics", "biology", "history", "geography"),
    9: ("chinese", "math", "english", "physics", "history", "chemistry"),
}


def validate_grade(grade: int) -> int:
    if grade not in _CURRICULUM:
        raise ValueError("年级必须是一年级到九年级")
    return grade


def validate_semester(semester: str) -> str:
    value = semester.strip().lower()
    if value not in SEMESTER_LABELS:
        raise ValueError("学期必须是 upper 或 lower")
    return value


def grade_label(grade: int) -> str:
    return _GRADE_CN[validate_grade(grade)]


def semester_label(semester: str) -> str:
    return SEMESTER_LABELS[validate_semester(semester)]


def profile_grade_label(grade: int, semester: str) -> str:
    return f"{grade_label(grade)}{semester_label(semester)}"


def allowed_subjects(grade: int) -> tuple[str, ...]:
    return _CURRICULUM[validate_grade(grade)]


def subject_payload(subject: str) -> dict[str, str]:
    cfg = SUBJECTS[subject]
    return {"key": subject, "label": cfg["label"], "icon": cfg["icon"]}


def textbook_filename(grade: int, semester: str, subject: str) -> str:
    validate_grade(grade)
    validate_semester(semester)
    if subject not in allowed_subjects(grade):
        raise ValueError(f"{grade_label(grade)}没有{SUBJECTS.get(subject, {}).get('label', subject)}教材")
    return f"{SUBJECTS[subject]['label']}_{grade_label(grade)}_{semester_label(semester)}.pdf"


def textbook_id(grade: int, semester: str, subject: str) -> str:
    textbook_filename(grade, semester, subject)
    return f"grade_{grade:02d}_{validate_semester(semester)}_{subject}"


def textbook_path(root: Path, grade: int, semester: str, subject: str) -> Path:
    return root / grade_label(grade) / textbook_filename(grade, semester, subject)
