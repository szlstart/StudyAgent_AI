from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import ClientOSError

from src.agents.homework.grade_agent import GradeAgent
from src.agents.homework.knowpoint_agent import KnowPointAgent
from src.agents.homework.models import (
    ErrorType,
    ExtractedQuestion,
    GradeResult,
    GradedQuestion,
    HomeworkResult,
    QuestionType,
)
from src.agents.homework.ocr_agent import OCRAgent, OCRPageProcessingError
from src.agents.homework.wrong_book_service import WrongBookService
from src.agents.memory.memory_service import MemoryService
from src.logging import get_logger
from src.services.question_bank import QuestionBankService
from src.services.llm.exceptions import LLMAPIError
from src.services.llm.factory import _is_transient_gateway_model_error
from src.services.llm.error_mapping import map_error
from src.services.user_files import resolve_user_file


def _question(number: str) -> ExtractedQuestion:
    return ExtractedQuestion(
        number=number,
        question_text=f"题目 {number}",
        student_answer="答案",
        question_type=QuestionType.SHORT_ANSWER,
    )


def test_grade_results_use_stable_ids_when_model_reorders_rows():
    agent = object.__new__(GradeAgent)
    agent.logger = get_logger("grade-test")
    raw = json.dumps({"results": [
        {"item_id": "q2", "number": "2", "grade": "wrong", "brief_comment": "第二题"},
        {"item_id": "q1", "number": "1", "grade": "correct", "brief_comment": "第一题"},
    ]})
    results = agent._parse_and_merge(raw, [_question("1"), _question("2")])
    assert [item.grade for item in results] == [GradeResult.CORRECT, GradeResult.WRONG]
    assert [item.brief_comment for item in results] == ["第一题", "第二题"]


def test_grade_missing_middle_row_never_shifts_later_grade():
    agent = object.__new__(GradeAgent)
    agent.logger = get_logger("grade-test")
    raw = json.dumps({"results": [
        {"item_id": "q1", "number": "1", "grade": "correct"},
        {"item_id": "q3", "number": "3", "grade": "wrong"},
    ]})
    results = agent._parse_and_merge(raw, [_question("1"), _question("2"), _question("3")])
    assert [item.grade for item in results] == [
        GradeResult.CORRECT, GradeResult.SKIP, GradeResult.WRONG,
    ]


def test_knowpoint_uses_stable_id_with_duplicate_question_numbers():
    first = GradedQuestion(
        number="(1)", question_text="甲", student_answer="", question_type=QuestionType.UNKNOWN,
        correct_answer="", grade=GradeResult.SKIP,
    )
    second = first.model_copy(update={"question_text": "乙"})
    agent = object.__new__(KnowPointAgent)
    agent.logger = get_logger("knowpoint-test")
    agent._apply_annotations([first, second], [
        {"item_id": "q2", "number": "(1)", "knowledge_points": ["乙知识点"], "difficulty": "hard"},
        {"item_id": "q1", "number": "(1)", "knowledge_points": ["甲知识点"], "difficulty": "easy"},
    ])
    assert first.knowledge_points == ["甲知识点"]
    assert second.knowledge_points == ["乙知识点"]


def test_question_bank_key_preserves_math_operators():
    assert QuestionBankService._normalize("计算 2 + 3") != QuestionBankService._normalize("计算 2 - 3")
    assert QuestionBankService._make_key(QuestionBankService._normalize("2+3")) != (
        QuestionBankService._make_key(QuestionBankService._normalize("2-3"))
    )


def test_question_bank_direct_grading_is_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("QUESTION_BANK_DIRECT_GRADING", raising=False)
    bank = QuestionBankService(tmp_path)
    result = asyncio.run(bank.lookup("下列选项中正确的是 A 选项", "A", "choice", "math"))
    assert result is None


def test_corrupt_memory_and_wrongbook_fail_closed(tmp_path):
    memory = MemoryService(tmp_path)
    memory._memory_file.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="停止写入"):
        asyncio.run(memory.update_preferences(nickname="不应覆盖"))
    assert memory._memory_file.read_text(encoding="utf-8") == "{broken"

    wrong = WrongBookService(tmp_path)
    wrong._path.parent.mkdir(parents=True, exist_ok=True)
    wrong._path.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="停止写入"):
        asyncio.run(wrong.list_entries())
    assert wrong._path.read_text(encoding="utf-8") == "not-json"


def test_persisted_user_file_path_cannot_escape_user_root(tmp_path):
    user_root = tmp_path / "user"
    user_root.mkdir()
    inside = user_root / "image.png"
    inside.write_bytes(b"image")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    assert resolve_user_file(inside, user_root) == inside.resolve()
    assert resolve_user_file(outside, user_root) is None


def test_mastered_wrong_question_is_removed_from_current_profile(tmp_path):
    async def scenario():
        result = HomeworkResult(
            subject="math", questions=[GradedQuestion(
                number="1", question_text="2+2", student_answer="5",
                question_type=QuestionType.CALCULATION, correct_answer="4",
                grade=GradeResult.WRONG, error_type=ErrorType.CALCULATION_ERROR,
                knowledge_points=["整数加法"],
            )],
        )
        result.compute_stats()
        wrong = WrongBookService(tmp_path)
        await wrong.save_from_result(result)
        memory = MemoryService(tmp_path)
        entries = await wrong.list_entries(mastered=False)
        await memory.reconcile_learning_state(entries)
        assert (await memory.load_memory()).knowledge_gaps["math"] == ["整数加法"]

        await wrong.mark_mastered(entries[0].entry_id, True)
        await memory.reconcile_learning_state(await wrong.list_entries(mastered=False))
        current = await memory.load_memory()
        assert current.knowledge_gaps == {}
        assert current.error_pattern_counts == {}

    asyncio.run(scenario())


def test_ocr_reports_failed_page_instead_of_silently_skipping():
    class FakeOCR(OCRAgent):
        def __init__(self):
            self.logger = get_logger("ocr-test")

        async def _process_single_image(self, image, subject, page_index):
            if page_index == 1:
                raise RuntimeError("upstream failed")
            return [_question(str(page_index + 1)).model_copy(update={"page_index": page_index})]

    with pytest.raises(OCRPageProcessingError, match="第 2 页"):
        asyncio.run(FakeOCR().process(["a", "b"], "math"))


def test_ocr_progress_distinguishes_success_from_failure():
    events = []

    class FakeOCR(OCRAgent):
        def __init__(self):
            self.logger = get_logger("ocr-progress-test")

        async def _process_single_image(self, image, subject, page_index):
            if page_index == 1:
                raise RuntimeError("upstream failed")
            return [_question("20").model_copy(update={"page_index": page_index})]

    with pytest.raises(OCRPageProcessingError):
        asyncio.run(FakeOCR().process(
            ["a", "b"], "math",
            progress_callback=lambda done, total, page, ok: events.append(
                (done, total, page, ok)
            ),
        ))
    assert events == [(1, 2, 0, True), (2, 2, 1, False)]


def test_ocr_does_not_report_model_outage_as_bad_image():
    class FakeOCR(OCRAgent):
        def __init__(self):
            self.logger = get_logger("ocr-model-outage-test")

        async def _process_single_image(self, image, subject, page_index):
            raise LLMAPIError(
                "upstream model channel unavailable",
                status_code=503,
                provider="openai",
            )

    with pytest.raises(LLMAPIError, match="model channel unavailable"):
        asyncio.run(FakeOCR().process(["clear-page-1", "clear-page-2"], "math"))


def test_only_apinebula_transient_model_channel_404_is_retried():
    error = LLMAPIError(
        'The model `gpt-5.5` does not exist or you do not have access to it; model_not_found',
        status_code=404,
        provider="openai",
    )
    assert _is_transient_gateway_model_error(error, "https://apinebula.ai/v1")
    assert not _is_transient_gateway_model_error(error, "https://api.openai.com/v1")
    assert not _is_transient_gateway_model_error(
        LLMAPIError("ordinary missing route", status_code=404),
        "https://apinebula.ai/v1",
    )


def test_asyncio_timeout_maps_to_retriable_timeout_error():
    mapped = map_error(asyncio.TimeoutError(), provider="openai")
    assert mapped.status_code == 408
    assert "响应超时" in str(mapped)


def test_connection_reset_maps_to_retriable_service_error():
    mapped = map_error(ClientOSError(54, "Connection reset by peer"), provider="openai")
    assert mapped.status_code == 503
    assert "连接中断" in str(mapped)


def test_ocr_rejects_empty_json_placeholder():
    agent = object.__new__(OCRAgent)
    agent.logger = get_logger("ocr-empty-placeholder-test")

    assert agent._parse_response('{"status":"unable to structure"}', 0) == []


def test_ocr_accepts_json_object_questions_contract():
    agent = object.__new__(OCRAgent)
    agent.logger = get_logger("ocr-json-contract-test")
    raw = json.dumps({"questions": [{
        "number": "21(1)",
        "question_text": "求 y 与 x 之间的解析式",
        "student_answer": "y=4x",
        "question_type": "calculation",
        "score_value": None,
        "visual_context": "分段函数图像",
    }]}, ensure_ascii=False)

    questions = agent._parse_response(raw, 0)
    assert len(questions) == 1
    assert questions[0].question_text == "求 y 与 x 之间的解析式"


def test_ocr_sorts_shuffled_pages_by_printed_major_and_sub_numbers():
    questions = [
        ExtractedQuestion(number="22(2)", major_number="22", sub_number="(2)", question_text="乙", page_index=0),
        ExtractedQuestion(number="20(1)", major_number="20", sub_number="(1)", question_text="甲", page_index=2),
        ExtractedQuestion(number="23(1)", major_number="23", sub_number="(1)", question_text="丁", page_index=1),
        ExtractedQuestion(number="22(1)", major_number="22", sub_number="(1)", question_text="丙", page_index=0),
    ]

    ordered = OCRAgent._sort_by_printed_number(questions)

    assert [question.number for question in ordered] == [
        "20(1)", "22(1)", "22(2)", "23(1)",
    ]


def test_grade_preserves_structure_and_freeform_explanation_fields():
    question = ExtractedQuestion(
        number="20(1)", major_number="20", sub_number="(1)",
        stem_text="矩形 ABCD 中两点运动。", subquestion_text="求 PQ 的长。",
        question_text="矩形 ABCD 中两点运动。\n求 PQ 的长。",
        student_answer="4√2", question_type=QuestionType.CALCULATION,
    )
    agent = object.__new__(GradeAgent)
    agent.logger = get_logger("grade-rich-result-test")
    raw = json.dumps({"results": [{
        "item_id": "q1", "number": "20(1)", "grade": "correct",
        "correct_answer": "4√2", "brief_comment": "作辅助线的方法正确",
        "solution_steps": "过点P作PM垂直BC，再用勾股定理。",
        "error_reason": "",
    }]}, ensure_ascii=False)

    result = agent._parse_and_merge(raw, [question])[0]

    assert (result.major_number, result.sub_number) == ("20", "(1)")
    assert result.solution_steps == "过点P作PM垂直BC，再用勾股定理。"
    assert result.correct_answer == result.student_answer == "4√2"
