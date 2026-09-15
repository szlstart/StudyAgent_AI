"""Read-only official textbook API scoped to the logged-in school profile."""

from __future__ import annotations

import asyncio
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from src.agents.explain.kb_manager import TextbookKBManager
from src.api.dependencies import current_user
from src.domain.curriculum import allowed_subjects, textbook_id as curriculum_textbook_id
from src.services.auth import CurrentUser

router = APIRouter(prefix="/api/v1/textbook", tags=["textbook"])
_manager = TextbookKBManager()
_index_tasks: dict[str, asyncio.Task] = {}


@router.get("/status")
async def get_status(user: CurrentUser = Depends(current_user)) -> dict:
    subjects = allowed_subjects(user.grade)
    statuses = await asyncio.gather(*(
        asyncio.to_thread(_manager.status, user.grade, user.semester, subject)
        for subject in subjects
    ))
    return {
        "grade": user.grade,
        "semester": user.semester,
        "grade_label": user.to_dict()["grade_label"],
        "textbooks": list(statuses),
    }


async def _index_background(grade: int, semester: str, subject: str) -> None:
    await _manager.reindex(grade, semester, subject)


@router.post("/reindex/{subject}")
async def reindex_subject(subject: str, user: CurrentUser = Depends(current_user)) -> dict:
    subject = subject.strip().lower()
    if subject not in allowed_subjects(user.grade):
        raise HTTPException(status_code=400, detail="该科目不属于当前年级")
    if not _manager.source_path(user.grade, user.semester, subject).exists():
        raise HTTPException(status_code=404, detail="教材 PDF 不存在")
    tid = curriculum_textbook_id(user.grade, user.semester, subject)
    running = _index_tasks.get(tid)
    if running and not running.done():
        return {"ok": True, "message": "索引任务已在运行，请勿重复提交", "textbook_id": tid}
    task = asyncio.create_task(_index_background(user.grade, user.semester, subject))
    _index_tasks[tid] = task
    task.add_done_callback(lambda finished, key=tid: _index_tasks.pop(key, None))
    return {"ok": True, "message": "索引任务已启动", "textbook_id": tid}


@router.get("/file/{subject}")
@router.get("/file/{subject}/{filename}")
async def view_textbook_file(
    subject: str,
    filename: str | None = None,
    textbook: str | None = Query(default=None),
    user: CurrentUser = Depends(current_user),
):
    subject = subject.strip().lower()
    if subject not in allowed_subjects(user.grade):
        raise HTTPException(status_code=404, detail="该科目不属于当前年级")
    expected_textbook = curriculum_textbook_id(user.grade, user.semester, subject)
    if textbook is not None and textbook != expected_textbook:
        raise HTTPException(status_code=409, detail="登录年级或学期已变化，请刷新教材列表")
    path = _manager.source_path(user.grade, user.semester, subject)
    if not path.exists():
        raise HTTPException(status_code=404, detail="教材 PDF 不存在")
    if filename is not None and filename != path.name:
        raise HTTPException(status_code=409, detail="教材文件名与当前登录档案不匹配，请刷新教材列表")
    encoded = quote(path.name, safe="")
    return FileResponse(
        str(path), media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{encoded}",
            "Cache-Control": "private, no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Vary": "Cookie",
            "X-StudyBuddy-Textbook": expected_textbook,
        },
    )
