"""
History API Router
==================

路由前缀：/api/v1/history

Endpoints
---------
GET  /api/v1/history/homework          — 作业批改历史列表（倒序）
GET  /api/v1/history/homework/{id}     — 单次批改详情
GET  /api/v1/history/chat              — 聊天会话列表
GET  /api/v1/history/chat/{id}         — 单次会话详情
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.logging import get_logger
from src.api.dependencies import current_user
from src.services.auth import CurrentUser, user_data_dir

logger = get_logger("History")

router = APIRouter()

_SUBJECT_CN = {
    "math": "数学", "physics": "物理", "chemistry": "化学",
    "english": "英语", "chinese": "语文", "history": "历史", "biology": "生物",
    "geography": "地理",
}


def _homework_dir(user_id: str) -> Path:
    return user_data_dir(user_id) / "history" / "homework"


def _sessions_dir(user_id: str) -> Path:
    return user_data_dir(user_id) / "memory" / "sessions"


def _safe_record_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", value):
        raise HTTPException(status_code=400, detail="记录 ID 格式不正确")
    return value


def _attach_legacy_homework_images(data: dict, user_id: str) -> None:
    """Recover private page images for records created before image URLs existed.

    New records persist exact crop URLs.  For legacy records only, choose the
    nearest preceding upload batch of the same subject and expected page count.
    A narrow two-hour window avoids ever attaching an unrelated homework set.
    """
    questions = data.get("questions") or []
    if not questions or any(q.get("question_image_urls") for q in questions):
        return
    subject = str(data.get("subject") or "")
    expected = int(data.get("image_count") or 0)
    try:
        graded_at = datetime.fromisoformat(str(data.get("graded_at")))
    except (TypeError, ValueError):
        return
    image_dir = user_data_dir(user_id) / "homework_images"
    pattern = re.compile(
        rf"^(\d{{8}}_\d{{6}})_[0-9a-f]{{8}}_{re.escape(subject)}_p(\d+)\.(?:jpg|jpeg|png|webp)$",
        re.IGNORECASE,
    )
    batches: dict[str, dict[int, Path]] = {}
    for path in image_dir.iterdir() if image_dir.is_dir() else []:
        match = pattern.match(path.name)
        if match:
            batch = path.name.rsplit("_p", 1)[0]
            batches.setdefault(batch, {})[int(match.group(2)) - 1] = path
    choices: list[tuple[float, dict[int, Path]]] = []
    for batch, pages in batches.items():
        if expected and len(pages) < expected:
            continue
        try:
            uploaded_at = datetime.strptime(batch[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        delta = (graded_at.replace(tzinfo=None) - uploaded_at).total_seconds()
        if 0 <= delta <= 2 * 60 * 60:
            choices.append((delta, pages))
    if not choices:
        return
    pages = min(choices, key=lambda item: item[0])[1]
    for question in questions:
        page = pages.get(int(question.get("page_index") or 0))
        if page:
            question["question_image_urls"] = [f"/api/v1/homework/images/{page.name}"]


# ─────────────────────────────────────────────────────────────────────────────
# Homework history
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/homework")
async def list_homework_history(
    subject: Optional[str] = Query(default=None, description="按科目过滤"),
    limit: int = Query(default=30, ge=1, le=100),
    user: CurrentUser = Depends(current_user),
):
    """
    返回作业批改历史列表（最新在前）。

    每条记录包含：id、科目、批改时间、题目总数、正确率等摘要信息。
    """
    history_dir = _homework_dir(user.id)
    history_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(history_dir.glob("*.json"), reverse=True)
    results = []
    for f in files:
        # Per-question tutoring threads are restored inside the right-side
        # study dock. Keep them out of the general "题目询问" history so dozens
        # of question-scoped sessions do not make that list noisy.
        if f.stem.startswith("sq_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if subject and data.get("subject") != subject:
                continue
            total = data.get("total_questions", 0)
            correct = data.get("correct_count", 0)
            results.append({
                "id":              f.stem,
                "subject":         data.get("subject", ""),
                "subject_cn":      _SUBJECT_CN.get(data.get("subject", ""), data.get("subject", "")),
                "graded_at":       data.get("graded_at", ""),
                "total_questions": total,
                "correct_count":   correct,
                "wrong_count":     data.get("wrong_count", 0),
                "partial_count":   data.get("partial_count", 0),
                "blank_count":     data.get("blank_count", 0),
                "accuracy_rate":   round(correct / total, 3) if total > 0 else 0.0,
                "exam_tags":       data.get("exam_tags", []),
                "weak_knowledge_points": data.get("weak_knowledge_points", []),
            })
            if len(results) >= limit:
                break
        except Exception as e:
            logger.warning(f"[History] 读取批改记录失败 {f.name}: {e}")

    return results


@router.get("/homework/{record_id}")
async def get_homework_detail(record_id: str, user: CurrentUser = Depends(current_user)):
    """返回单次作业批改的完整详情（含所有题目批改结果）。"""
    record_id = _safe_record_id(record_id)
    path = _homework_dir(user.id) / f"{record_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="批改记录不存在")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _attach_legacy_homework_images(data, user.id)
        return data
    except Exception as e:
        logger.error(f"[History] 读取详情失败 {record_id}: {e}")
        raise HTTPException(status_code=500, detail="读取记录失败")


# ─────────────────────────────────────────────────────────────────────────────
# Chat session history
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/chat")
async def list_chat_sessions(
    subject: Optional[str] = Query(default=None, description="按科目过滤"),
    limit: int = Query(default=30, ge=1, le=100),
    user: CurrentUser = Depends(current_user),
):
    """
    返回聊天会话历史列表（最新在前）。

    每条记录包含：session_id、科目、最后更新时间、轮数、摘要片段。
    """
    sessions_dir = _sessions_dir(user.id)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        sessions_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True,
    )
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            subj = data.get("subject", "")
            if subject and subj != subject:
                continue
            summary = data.get("compressed_summary", "")
            messages = data.get("messages", [])
            # last user message as preview
            preview = ""
            for m in reversed(messages):
                if m.get("role") == "user" and m.get("content", "").strip():
                    preview = m["content"][:80]
                    break
            results.append({
                "session_id":   data.get("session_id", f.stem),
                "subject":      subj,
                "subject_cn":   _SUBJECT_CN.get(subj, subj),
                "turn_count":   data.get("turn_count", 0),
                "last_updated": data.get("last_updated_at", data.get("created_at", "")),
                "has_summary":  bool(summary),
                "preview":      preview or (summary[:80] if summary else ""),
            })
            if len(results) >= limit:
                break
        except Exception as e:
            logger.warning(f"[History] 读取会话记录失败 {f.name}: {e}")

    return results


@router.get("/chat/{session_id}")
async def get_chat_session(session_id: str, user: CurrentUser = Depends(current_user)):
    """返回单次聊天会话的完整记录（含所有消息）。"""
    session_id = _safe_record_id(session_id)
    path = _sessions_dir(user.id) / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="会话记录不存在")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"[History] 读取会话失败 {session_id}: {e}")
        raise HTTPException(status_code=500, detail="读取会话失败")
