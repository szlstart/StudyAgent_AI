from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.agents.memory.profile_schema import ChatSession
from src.api.routers import explain as explain_router
from src.api.routers.explain import StudyChatRequest, StudyQuestionContext
from src.services.auth import CurrentUser


def _user() -> CurrentUser:
    return CurrentUser(
        id="student-1", name="小明", grade=8, semester="lower",
        created_at="2026-09-10", updated_at="2026-09-10",
    )


def test_question_study_chat_streams_and_persists_one_turn(monkeypatch):
    recorded: list[dict] = []

    class FakeMemory:
        async def load_session(self, session_id: str):
            return ChatSession(session_id=session_id, subject="math")

    class FakeAgent:
        async def retrieve_snippets(self, *args, **kwargs):
            return ["[第 131 页] 一次函数"]

        async def stream_llm(self, *args, **kwargs):
            yield "先看斜率，"
            yield "再确定定义域。"

    async def fake_memory_prompt(*args, **kwargs):
        return "按八年级学生偏好清晰讲解。"

    async def fake_record(*args, **kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr(explain_router, "_agent", FakeAgent())
    monkeypatch.setattr(explain_router, "_memory", lambda _user_id: FakeMemory())
    monkeypatch.setattr(explain_router, "_memory_prompt", fake_memory_prompt)
    monkeypatch.setattr(explain_router, "_record_learning_turn", fake_record)

    body = StudyChatRequest(
        session_id="sq_math21_abc123",
        subject="math",
        message="为什么第二段要写上限？",
        context=StudyQuestionContext(
            source="wrongbook",
            title="第 21 题 · 小问（1）",
            question_text="根据图像写出分段函数解析式。",
            student_answer="第二段只写了 x>2",
            correct_answer="第二段应为 2<x≤18",
            knowledge_points=["分段函数"],
        ),
    )

    async def scenario():
        response = await explain_router.stream_study_chat(body, _user())
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return [json.loads(line) for line in "".join(chunks).splitlines()]

    events = asyncio.run(scenario())
    assert [event["type"] for event in events] == ["meta", "delta", "delta", "done"]
    assert events[0]["used_rag"] is True
    assert recorded[0]["session_id"] == "sq_math21_abc123"
    assert recorded[0]["assistant_text"] == "先看斜率，再确定定义域。"


def test_study_chat_context_rejects_invalid_session_id():
    try:
        StudyChatRequest(
            session_id="../../other-user",
            subject="math",
            message="解释一下",
            context=StudyQuestionContext(
                source="homework", question_text="1+1=?",
            ),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe session id must be rejected")


def test_study_chat_image_is_limited_to_current_user_directory(tmp_path: Path, monkeypatch):
    user_root = tmp_path / "student-1"
    image_dir = user_root / "homework_images"
    image_dir.mkdir(parents=True)
    (image_dir / "question.jpg").write_bytes(b"safe-image")
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"secret")
    monkeypatch.setattr(explain_router, "user_data_dir", lambda _user_id: user_root)

    safe = explain_router._question_image_data_url(
        _user(), "/api/v1/homework/images/question.jpg"
    )
    assert safe.startswith("data:image/jpeg;base64,")
    assert explain_router._question_image_data_url(_user(), "https://evil.invalid/a.jpg") == ""
    assert explain_router._question_image_data_url(
        _user(), "/api/v1/homework/images/../secret.jpg"
    ) == ""
