from pathlib import Path

from PIL import Image

from src.agents.homework.models import (
    GradedQuestion,
    GradeResult,
    QuestionType,
    WrongBookEntry,
)
from src.agents.homework.ocr_agent import OCRAgent
from src.agents.homework.grade_agent import _COMMON_OUTPUT_RULES, _normalize_math_markup
from src.api.routers.homework import _attach_textbook_references, _textbook_reference
from src.api.routers.wrong_book import WrongBookPublicEntry
from src.services.homework_images import attach_question_crops, create_question_crops


def _graded(number: str, bbox: list[int]) -> GradedQuestion:
    return GradedQuestion(
        number=number,
        major_number="20",
        sub_number=f"({number[-1]})",
        stem_text="如图，已知完整条件。",
        subquestion_text=f"求第 {number[-1]} 小问。",
        question_text=f"如图，已知完整条件。求第 {number[-1]} 小问。",
        student_answer="过程",
        question_type=QuestionType.CALCULATION,
        page_index=0,
        bounding_box=bbox,
        correct_answer="答案",
        grade=GradeResult.WRONG,
    )


def test_major_question_crop_unions_all_subquestions(tmp_path: Path):
    source = tmp_path / "page.jpg"
    Image.new("RGB", (1000, 1000), "white").save(source)
    questions = [_graded("20(1)", [100, 100, 600, 400]), _graded("20(2)", [100, 380, 700, 800])]

    crops = create_question_crops(questions, [str(source)], "batch")
    attach_question_crops(questions, crops)

    crop_path = Path(questions[0].question_image_paths[0])
    assert crop_path.is_file()
    assert questions[0].question_image_urls == questions[1].question_image_urls
    with Image.open(crop_path) as image:
        assert image.width >= 640
        assert image.height >= 740


def test_missing_box_falls_back_to_complete_page(tmp_path: Path):
    source = tmp_path / "page.jpg"
    Image.new("RGB", (800, 600), "white").save(source)
    questions = [_graded("20(1)", [100, 100, 600, 400]), _graded("20(2)", [])]
    crops = create_question_crops(questions, [str(source)], "fallback")
    with Image.open(crops["major-20"][0]) as image:
        assert image.size == (800, 600)


def test_textbook_references_are_extracted_from_real_snippets():
    snippets = ["[第 131 页] 第二十三章　一次函数\n本章学习函数图像。"]
    question = _graded("20(1)", [])
    question.question_text = "求一次函数的解析式与函数图像"
    _attach_textbook_references([question], snippets)
    assert _textbook_reference(snippets[0]) == "第二十三章 一次函数 · 第 131 页"
    assert question.textbook_refs == ["第二十三章 一次函数 · 第 131 页"]


def test_wrongbook_public_model_does_not_serialize_absolute_path():
    entry = WrongBookEntry(
        entry_id="math_batch_p1_20",
        subject="math",
        question_text="题目",
        student_answer="错答",
        correct_answer="正答",
        question_type=QuestionType.CALCULATION,
        grade=GradeResult.WRONG,
        source_image_path="/Users/student/private/homework.jpg",
    )
    public = WrongBookPublicEntry.model_validate(entry.model_dump())
    assert "source_image_path" not in public.model_dump()


def test_ocr_box_validation_rejects_unsafe_shapes():
    assert OCRAgent._safe_bounding_box([100, 100, 600, 700]) == [100, 100, 600, 700]
    assert OCRAgent._safe_bounding_box([1, 2, 3, 4]) == []
    assert OCRAgent._safe_bounding_box([1, 2, 3]) == []


def test_grading_contract_requires_readable_steps_and_valid_math_markup():
    assert "禁止把整段推导挤成一行" in _COMMON_OUTPUT_RULES
    assert "行内公式用 $...$" in _COMMON_OUTPUT_RULES
    assert "中文讲解放在公式外" in _COMMON_OUTPUT_RULES


def test_grade_output_normalizes_bare_and_escaped_latex():
    bare = r"y=\begin{cases}4x, 0\le x\le 2,\\-\frac12x+9, x>2\end{cases}"
    normalized = _normalize_math_markup(bare, 8_000)
    assert normalized.startswith("$$y=\\begin{cases}")
    assert normalized.endswith("\\end{cases}$$")

    already_delimited = "$$y=\\begin{cases}4x,\\\\x<18\\n\\end{cases}$$"
    assert _normalize_math_markup(already_delimited, 8_000) == already_delimited
    assert _normalize_math_markup(r"$2\<x\le18$", 8_000) == r"$2<x\le18$"


def test_homework_workspace_uses_rich_math_renderer():
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text(encoding="utf-8")
    assert "function renderStudyContent" in html
    assert "function renderStudyFormula" in html
    assert "wrapBareLatexEnvironments(formatStudyText(text, mode))" in html
    assert "value.replace(`STUDYEXISTINGMATH${index}Z`, () => formula)" in html
    assert "marked.parse(escapeHtml(protectedText)" in html
    assert "rendered.replaceAll(token, renderStudyFormula(formula))" in html
    assert "class=\"study-rich\"" in html
    assert "studyMd(q.solution_steps" in html
    assert ".study-rich ol { list-style:decimal outside; }" in html


def test_each_question_has_a_collapsible_streaming_study_chat_entry():
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text(encoding="utf-8")
    assert html.count('class="ask-ai-entry"') == 2
    assert 'class="study-dock"' in html
    assert "async studyChatSend()" in html
    assert "res.body.getReader()" in html
    assert "/api/v1/explain/study-chat/stream" in html
    assert "ui.navCollapsed=!ui.navCollapsed" in html


def test_original_screenshots_open_in_an_in_page_comparison_viewer():
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text(encoding="utf-8")
    assert "window.open(img" not in html
    assert html.count("openImageViewer(img") == 2
    assert 'class="image-compare-viewer"' in html
    assert "imageViewerZoom(delta)" in html
    assert "imageViewer.rotation=(imageViewer.rotation+90)%360" in html
    assert "答案与解析区域仍可滚动查看" in html
