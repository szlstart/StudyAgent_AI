from __future__ import annotations

import asyncio
import io
import json

import pytest
from starlette.datastructures import Headers, UploadFile
from PIL import Image

from src.agents.homework.models import (
    ErrorType,
    ExtractedQuestion,
    GradeResult,
    GradedQuestion,
    HomeworkResult,
    QuestionType,
)
from src.agents.homework.wrong_book_service import WrongBookService
from src.agents.homework.knowpoint_agent import KnowPointAgent
from src.agents.memory.memory_service import MemoryService
from src.api.routers import homework as homework_router
from src.services.auth import CurrentUser
from src.services.question_bank import QuestionBankService


def _user() -> CurrentUser:
    return CurrentUser(
        id="manual-contract-user",
        name="测试学生",
        grade=7,
        semester="lower",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def test_recent_chat_turns_are_available_before_summary_threshold(tmp_path):
    async def scenario():
        service = MemoryService(tmp_path)
        long_answer = "完整讲解" * 300
        await service.record_turn(
            "sess_context", "math", "上一题为什么这样列式？",
            "因为总量等于单价乘数量。" + long_answer,
        )
        context = await service.get_memory_context("math", "sess_context")
        assert "上一题为什么这样列式" in context.session_summary
        assert "单价乘数量" in context.session_summary
        session_before_summary = await service.load_session("sess_context")
        assert session_before_summary.messages[-1].content.endswith(long_answer)

        await service.save_compressed_summary("sess_context", "【学生提问】列式\n【学情分析】数量关系")
        session = await service.load_session("sess_context")
        assert session.turn_records == []
        assert "数量关系" in session.compressed_summary

    asyncio.run(scenario())


def test_session_id_cannot_escape_the_user_directory(tmp_path):
    service = MemoryService(tmp_path)
    with pytest.raises(ValueError, match="会话 ID"):
        asyncio.run(service.load_session("../../another-user"))


def test_concurrent_memory_writes_from_multiple_service_instances_are_not_lost(tmp_path):
    async def scenario():
        first = MemoryService(tmp_path)
        second = MemoryService(tmp_path)
        await asyncio.gather(*(
            (first if i % 2 else second).record_turn(
                "shared_session", "math", f"问题{i}", f"回答{i}"
            )
            for i in range(20)
        ))
        session = await first.load_session("shared_session")
        assert session.turn_count == 20
        assert len(session.messages) == 40
        assert len(session.turn_records) == 20

    asyncio.run(scenario())


def test_question_bank_requires_two_consistent_model_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("QUESTION_BANK_DIRECT_GRADING", "1")
    question = GradedQuestion(
        number="1", question_text="下列整数加法结果正确的是哪一项",
        student_answer="A", question_type=QuestionType.CHOICE,
        correct_answer="A", grade=GradeResult.CORRECT,
    )

    async def scenario():
        bank = QuestionBankService(tmp_path)
        assert await bank.save_batch([question], "math") == 1
        assert await bank.lookup(
            question.question_text, "A", "choice", "math"
        ) is None
        assert await bank.save_batch([question], "math") == 0
        hit = await bank.lookup(question.question_text, "B", "choice", "math")
        assert hit is not None
        assert hit["grade"] == "wrong"
        assert hit["error_type"] == "other"

    asyncio.run(scenario())


def test_wrong_book_preserves_curriculum_and_knowledge_point(tmp_path):
    async def scenario():
        result = HomeworkResult(
            subject="math",
            grade_level=7,
            semester="lower",
            textbook_id="grade_07_lower_math",
            questions=[GradedQuestion(
                number="1",
                question_text="2x=6，求x",
                student_answer="2",
                question_type=QuestionType.CALCULATION,
                correct_answer="3",
                grade=GradeResult.WRONG,
                error_type=ErrorType.CALCULATION_ERROR,
                knowledge_points=["一元一次方程"],
                difficulty="easy",
            )],
        )
        result.compute_stats()
        service = WrongBookService(tmp_path)
        assert await service.save_from_result(result) == 1
        entry = (await service.list_entries())[0]
        assert (entry.grade_level, entry.semester) == (7, "lower")
        assert entry.textbook_id == "grade_07_lower_math"
        assert entry.knowledge_points == ["一元一次方程"]

    asyncio.run(scenario())


def test_knowpoint_accepts_json_object_wrapper():
    question = GradedQuestion(
        number="1", question_text="2+3=?", student_answer="5",
        question_type=QuestionType.CALCULATION, correct_answer="5",
        grade=GradeResult.CORRECT,
    )
    parsed = KnowPointAgent._annotation_items({
        "results": [{
            "number": "1", "knowledge_points": ["整数加法"],
            "difficulty": "easy", "is_weak_area": False,
        }],
    })
    agent = object.__new__(KnowPointAgent)
    agent._apply_annotations([question], parsed)
    assert question.knowledge_points == ["整数加法"]
    assert question.difficulty == "easy"


def test_homework_returns_fully_annotated_result_and_persists_before_response(
    tmp_path, monkeypatch,
):
    user_root = tmp_path / "users"

    def local_user_dir(user_id: str):
        path = user_root / user_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    class FakeOCR:
        async def process(self, images, subject, **kwargs):
            assert len(images) == 1
            return [ExtractedQuestion(
                number="1", question_text="2x=6，求x", student_answer="2",
                question_type=QuestionType.CALCULATION, page_index=0,
            )]

    class FakeGrade:
        async def process(self, questions, **kwargs):
            return [GradedQuestion(
                **questions[0].model_dump(), correct_answer="3",
                grade=GradeResult.WRONG,
                error_type=ErrorType.CALCULATION_ERROR,
                brief_comment="移项计算错误",
            )]

    class FakeKnowPoint:
        def __init__(self, **kwargs):
            pass

        async def process(self, questions, **kwargs):
            questions[0].knowledge_points = ["一元一次方程"]
            questions[0].difficulty = "easy"
            return questions

    class FakeExamTag:
        def __init__(self, **kwargs):
            pass

        async def process(self, questions, **kwargs):
            return ["基础运算"], ["一元一次方程"]

    class FakeBank:
        async def lookup(self, *args, **kwargs):
            return None

    class FakeKB:
        async def search(self, *args, **kwargs):
            return ["[七年级下册数学 · 第1页] 一元一次方程"]

    class FakeEverMemOS:
        async def search_context(self, **kwargs):
            return []

    async def no_background(**kwargs):
        return None

    monkeypatch.setattr(homework_router, "user_data_dir", local_user_dir)
    monkeypatch.setattr(homework_router, "build_homework_ocr_agent", lambda _key: FakeOCR())
    monkeypatch.setattr(homework_router, "GradeAgent", FakeGrade)
    monkeypatch.setattr(homework_router, "KnowPointAgent", FakeKnowPoint)
    monkeypatch.setattr(homework_router, "ExamTagAgent", FakeExamTag)
    monkeypatch.setattr(homework_router, "_get_question_bank", lambda _curriculum: FakeBank())
    monkeypatch.setattr(homework_router, "_kb_manager", FakeKB())
    monkeypatch.setattr(homework_router, "get_evermemos_service", lambda _uid: FakeEverMemOS())
    monkeypatch.setattr(homework_router, "_background_persist_memory", no_background)
    homework_router._mem_services.clear()

    async def scenario():
        image = io.BytesIO()
        Image.new("RGB", (8, 6), (40, 80, 120)).save(image, format="PNG")
        upload = UploadFile(
            io.BytesIO(image.getvalue()), filename="page.png",
            headers=Headers({"content-type": "image/png"}),
        )
        result = await homework_router.grade_homework(
            files=[upload], subject="math", record_type="homework",
            exam_name="", model_key="", user=_user(),
        )
        await asyncio.sleep(0)
        return result

    result = asyncio.run(scenario())
    assert result.used_rag is True
    assert result.exam_tags == ["基础运算"]
    assert result.weak_knowledge_points == ["一元一次方程"]
    assert result.questions[0].knowledge_points == ["一元一次方程"]

    user_dir = local_user_dir(_user().id)
    wrong_entries = json.loads(
        (user_dir / "wrong_book" / "entries.json").read_text(encoding="utf-8")
    )
    assert wrong_entries[0]["knowledge_points"] == ["一元一次方程"]
    assert wrong_entries[0]["textbook_id"] == "grade_07_lower_math"

    history_files = list((user_dir / "history" / "homework").glob("*.json"))
    assert len(history_files) == 1
    history = json.loads(history_files[0].read_text(encoding="utf-8"))
    assert history["exam_tags"] == ["基础运算"]

    memory = json.loads((user_dir / "memory" / "user_memory.json").read_text(encoding="utf-8"))
    assert memory["performance_records"][-1]["weak_knowledge_points"] == ["一元一次方程"]
    assert memory["performance_records"][-1]["source_history_id"] == history_files[0].stem
