"""
Wrong Book API Router — 错题本
================================

Endpoints
---------
GET    /api/v1/wrong-book                    — 列出所有错题（支持 subject / mastered 过滤）
GET    /api/v1/wrong-book/stats              — 错题本统计摘要
GET    /api/v1/wrong-book/{id}               — 获取单条错题
POST   /api/v1/wrong-book/{id}/explain       — 一键解析该条错题（无需 body）
PUT    /api/v1/wrong-book/{id}/reviewed      — 标记已复习
PUT    /api/v1/wrong-book/{id}/mastered      — 设置掌握状态
DELETE /api/v1/wrong-book/{id}              — 删除一条错题
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

# ── Project root in path ──────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.explain import ExplainAgent, ExplainResponse
from src.agents.explain.explain_agent import ExplainRequest
from src.agents.homework.models import WrongBookEntry
from src.agents.homework.models import GradeResult
from src.agents.homework.wrong_book_service import WrongBookService
from src.agents.memory import MemoryService
from src.api.dependencies import current_user
from src.logging import get_logger
from src.services.auth import CurrentUser, user_data_dir
from src.services.evermemos import get_evermemos_service
from src.services.atomic_io import atomic_write_text
from src.services.user_files import resolve_user_file

logger = get_logger("WrongBook")

# ── Shared service instance（懒加载，首次调用时初始化）────────────────────────
_data_dir = _project_root / "data"
_services: dict[str, WrongBookService] = {}
_explain_agent: Optional[ExplainAgent] = None
_mem_services: dict[str, MemoryService] = {}


def get_service(user_id: str) -> WrongBookService:
    if user_id not in _services:
        _services[user_id] = WrongBookService(user_data_dir(user_id))
    return _services[user_id]


def _get_explain_agent() -> ExplainAgent:
    global _explain_agent
    if _explain_agent is None:
        _explain_agent = ExplainAgent(data_dir=_data_dir)
    return _explain_agent


def _get_mem_service(user_id: str) -> MemoryService:
    if user_id not in _mem_services:
        _mem_services[user_id] = MemoryService(user_data_dir(user_id))
    return _mem_services[user_id]


async def _reconcile_profile(user_id: str) -> None:
    """Keep current weakness counters aligned with the wrong-book backlog."""
    try:
        entries = await get_service(user_id).list_entries(mastered=False)
        await _get_mem_service(user_id).reconcile_learning_state(entries)
    except Exception as exc:
        logger.error(f"[WrongBook] 学生画像重算失败: {exc}", exc_info=True)


# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response helpers
# ─────────────────────────────────────────────────────────────────────────────

class MasteredUpdate(BaseModel):
    mastered: bool


class OverrideCorrectRequest(BaseModel):
    question_text: str = Field(min_length=1, max_length=12_000)
    subject: str = Field(min_length=1, max_length=30)
    student_answer: str = Field(default="", max_length=12_000)


class WrongBookStats(BaseModel):
    total: int
    mastered: int
    unmastered: int
    by_subject: dict[str, int]
    by_difficulty: dict[str, int]


class WrongBookPublicEntry(WrongBookEntry):
    """Browser-safe view: never expose the student's absolute local path."""
    source_image_path: Optional[str] = Field(default=None, exclude=True)


def _public_entry(entry: WrongBookEntry, user_id: str) -> WrongBookPublicEntry:
    payload = entry.model_dump()
    urls = list(entry.question_image_urls)
    if not urls and entry.source_image_path:
        path = resolve_user_file(entry.source_image_path, user_data_dir(user_id))
        if path and path.parent == user_data_dir(user_id) / "homework_images":
            urls = [f"/api/v1/homework/images/{path.name}"]
    payload["question_image_urls"] = urls
    return WrongBookPublicEntry.model_validate(payload)


def _adjust_latest_history(user_id: str, entry: WrongBookEntry) -> str | None:
    """把申诉结果同步到包含该错题的最近一份批改历史。"""
    history_dir = user_data_dir(user_id) / "history" / "homework"
    if not history_dir.exists():
        return None

    question_key = " ".join(entry.question_text.split()).casefold()
    answer_key = " ".join(entry.student_answer.split()).casefold()
    for path in sorted(history_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("subject") != entry.subject:
                continue
            matched = None
            for question in data.get("questions", []):
                if question.get("grade") not in {"wrong", "partial"}:
                    continue
                q_key = " ".join(str(question.get("question_text", "")).split()).casefold()
                a_key = " ".join(str(question.get("student_answer", "")).split()).casefold()
                if q_key == question_key and (not answer_key or a_key == answer_key):
                    matched = question
                    break
            if matched is None:
                continue

            matched["grade"] = "correct"
            if matched.get("score_value") is not None:
                matched["earned_score"] = matched["score_value"]
            matched["error_type"] = None
            matched["brief_comment"] = "已由学生申诉确认为正确"

            questions = data.get("questions", [])
            data["total_questions"] = len(questions)
            data["correct_count"] = sum(q.get("grade") == "correct" for q in questions)
            data["wrong_count"] = sum(q.get("grade") == "wrong" for q in questions)
            data["partial_count"] = sum(q.get("grade") == "partial" for q in questions)
            data["blank_count"] = sum(q.get("grade") == "blank" for q in questions)
            data["skip_count"] = sum(q.get("grade") == "skip" for q in questions)
            if questions and all(q.get("score_value") is not None for q in questions):
                data["total_score"] = sum(float(q["score_value"]) for q in questions)
                data["earned_score"] = sum(float(q.get("earned_score") or 0) for q in questions)
            atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
            return path.stem
        except Exception as exc:
            logger.warning(f"[WrongBook] 同步历史失败 {path.name}: {exc}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[WrongBookPublicEntry])
async def list_entries(
    subject: Optional[str] = Query(
        default=None,
        description="按科目过滤，如 math / physics / chinese",
    ),
    mastered: Optional[bool] = Query(
        default=None,
        description="true = 只看已掌握；false = 只看未掌握；不传 = 全部",
    ),
    user: CurrentUser = Depends(current_user),
):
    """
    列出错题本条目。

    - 不传任何参数 → 返回全部（按创建时间倒序）
    - `subject=math` → 只返回数学错题
    - `mastered=false` → 只返回尚未掌握的题目
    """
    svc = get_service(user.id)
    entries = await svc.list_entries(subject=subject, mastered=mastered)
    return [_public_entry(entry, user.id) for entry in entries]


@router.get("/stats", response_model=WrongBookStats)
async def get_stats(user: CurrentUser = Depends(current_user)):
    """
    错题本统计摘要：总数、已掌握数、按科目分布、按难度分布。
    """
    svc = get_service(user.id)
    stats = await svc.get_stats()
    return WrongBookStats(**stats)


@router.post("/{entry_id}/explain", response_model=ExplainResponse)
async def explain_entry(
    entry_id: str,
    user: CurrentUser = Depends(current_user),
):
    """
    **错题一键解析**：无需填写任何 body，系统自动加载错题上下文并生成详细解析。

    后端会自动读取该条错题的所有信息（题目、学生答案、正确答案、知识点、错误类型等），
    并以「错题解析模式」调用 AI 解析，重点分析：

    1. 正确答案的完整解题过程
    2. 学生答案为什么错（针对具体错误类型分析）
    3. 同类题型的解题技巧

    系统会按该错题保存的年级、学期和科目检索对应教材，确保解析在当前课本范围内。

    **调用示例**：
    ```
    POST /api/v1/wrong-book/english_20250228_143021_1/explain
    ```
    （无 request body）
    """
    # 1. 加载错题
    svc = get_service(user.id)
    entry = await svc.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"错题 '{entry_id}' 不存在")

    # 2. 将 WrongBookEntry 映射到 ExplainRequest（error_analysis 模式）
    request = ExplainRequest(
        question_text=entry.question_text,
        subject=entry.subject,
        student_answer=entry.student_answer,
        correct_answer=entry.correct_answer,
        question_type=entry.question_type.value,
        knowledge_points=list(entry.knowledge_points),
        error_type=entry.error_reason or (entry.error_type.value if entry.error_type else None),
        brief_comment=entry.brief_comment,
        mode="error_analysis",
    )

    # 解析也使用当前学生的讲解偏好、近期薄弱点和 EverOS 相关记忆。
    memory_context = (
        await _get_mem_service(user.id).get_memory_context(entry.subject)
    ).to_prompt_str()
    try:
        memories = await asyncio.wait_for(
            get_evermemos_service(user.id).search_context(
                subject=entry.subject,
                query=entry.question_text[:200],
                top_k=4,
            ),
            timeout=10,
        )
        if memories:
            memory_context += "\n\n【该学生的长期学习记忆】\n" + "\n".join(
                f"• {item}" for item in memories
            )
    except Exception as memory_exc:
        logger.debug(f"[WrongBook] EverMemOS 读取跳过: {memory_exc}")

    # 3. 尝试加载原始作业图片
    image_bytes: Optional[bytes] = None
    image_content_type = "image/jpeg"
    if entry.source_image_path:
        img_path = resolve_user_file(entry.source_image_path, user_data_dir(user.id))
        if img_path:
            try:
                image_bytes = img_path.read_bytes()
                # 根据文件后缀推断 MIME type
                sfx = img_path.suffix.lower()
                if sfx in (".png",):
                    image_content_type = "image/png"
                elif sfx in (".webp",):
                    image_content_type = "image/webp"
            except Exception as img_exc:
                logger.warning(f"[WrongBook] 读取图片失败（降级为纯文字解析）: {img_exc}")

    # 4. 解析
    try:
        # 错题解析固定使用 .env 中配置的系统模型（当前为 gpt-5.5），
        # 不允许通过查询参数切换到未配置的模型预设。
        explain_agent = _get_explain_agent()
        if image_bytes:
            logger.info(
                f"[WrongBook] 视觉解析 entry={entry_id} "
                f"image={len(image_bytes)//1024}KB q={entry.question_number!r}"
            )
            return await explain_agent.explain_with_image(
                request,
                image_bytes=image_bytes,
                question_number=entry.question_number or "",
                content_type=image_content_type,
                grade=entry.grade_level or user.grade,
                semester=entry.semester or user.semester,
                memory_context=memory_context,
            )
        return await explain_agent.explain(
            request, memory_context=memory_context,
            grade=entry.grade_level or user.grade,
            semester=entry.semester or user.semester,
        )
    except Exception as exc:
        logger.error(f"[WrongBook] 解析失败 entry={entry_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="错题解析服务暂时不可用，请稍后重试；如持续失败请查看服务日志",
        ) from exc


@router.get("/{entry_id}", response_model=WrongBookPublicEntry)
async def get_entry(entry_id: str, user: CurrentUser = Depends(current_user)):
    """获取指定 ID 的单条错题。"""
    svc = get_service(user.id)
    entry = await svc.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"错题 '{entry_id}' 不存在")
    return _public_entry(entry, user.id)


@router.put("/{entry_id}/reviewed", response_model=WrongBookPublicEntry)
async def mark_reviewed(entry_id: str, user: CurrentUser = Depends(current_user)):
    """
    标记此题已复习一次：review_count +1，last_reviewed_at 更新为当前时间。
    """
    svc = get_service(user.id)
    entry = await svc.mark_reviewed(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"错题 '{entry_id}' 不存在")
    return _public_entry(entry, user.id)


@router.put("/{entry_id}/mastered", response_model=WrongBookPublicEntry)
async def set_mastered(entry_id: str, body: MasteredUpdate, user: CurrentUser = Depends(current_user)):
    """
    设置掌握状态。

    - `{"mastered": true}` → 标记为已掌握
    - `{"mastered": false}` → 取消掌握（重新归入待复习）
    """
    svc = get_service(user.id)
    entry = await svc.mark_mastered(entry_id, body.mastered)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"错题 '{entry_id}' 不存在")
    await _reconcile_profile(user.id)
    if body.mastered:
        asyncio.create_task(
            get_evermemos_service(user.id).log_learning_correction(
                entry.subject, entry.question_text, "学生已掌握",
            )
        )
    return _public_entry(entry, user.id)


@router.post("/override-correct")
async def override_correct(body: OverrideCorrectRequest, user: CurrentUser = Depends(current_user)):
    """
    **学生申诉覆盖**：将某道被 AI 判为错误的题目标记为「实际正确」。

    操作流程：
    1. 按题目文字 + 科目 + 学生答案在错题本中查找对应条目
    2. 找到则删除该条目（题目从错题本移除）
    3. 调整该题对应的成绩与历史记录（错误/部分正确 -1，正确 +1）

    - 若未找到匹配条目，视为已经处理过，不重复调整成绩
    - 同时修正本地画像和对应的最近一份批改历史
    """
    svc = get_service(user.id)
    mem_svc = _get_mem_service(user.id)

    # 1. 查找错题条目
    entry_id = await svc.find_entry_by_question(
        question_text=body.question_text,
        subject=body.subject,
        student_answer=body.student_answer,
    )
    entry_snapshot = await svc.get_entry(entry_id) if entry_id else None

    # 2. 删除错题本条目（如果找到）
    deleted_id: Optional[str] = None
    if entry_id:
        deleted = await svc.delete_entry(entry_id)
        if deleted:
            deleted_id = entry_id
            logger.info(f"[WrongBook] 申诉覆盖：已删除错题 {entry_id}")

    # 3. 只有确实删除了错题时才调整一次，避免重复点击导致成绩反复累加。
    adjusted = False
    adjusted_history: str | None = None
    if deleted_id and entry_snapshot is not None:
        adjusted_history = _adjust_latest_history(user.id, entry_snapshot)
        adjusted = await mem_svc.adjust_last_record(
            subject=body.subject,
            delta_wrong=-1 if entry_snapshot.grade == GradeResult.WRONG else 0,
            delta_partial=-1 if entry_snapshot.grade == GradeResult.PARTIAL else 0,
            delta_correct=1,
            source_history_id=adjusted_history or "",
        )
        await _reconcile_profile(user.id)
        asyncio.create_task(
            get_evermemos_service(user.id).log_learning_correction(
                entry_snapshot.subject, entry_snapshot.question_text,
                "学生申诉后确认作答正确，旧错误判断作废",
            )
        )

    return {
        "ok": True,
        "deleted_entry_id": deleted_id,
        "adjusted_record": adjusted,
        "adjusted_history_id": adjusted_history,
        "already_applied": deleted_id is None,
    }


@router.delete("/{entry_id}")
async def delete_entry(entry_id: str, user: CurrentUser = Depends(current_user)):
    """
    永久删除一条错题。此操作不可恢复。
    """
    svc = get_service(user.id)
    deleted = await svc.delete_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"错题 '{entry_id}' 不存在")
    await _reconcile_profile(user.id)
    return {"deleted": True, "entry_id": entry_id}
