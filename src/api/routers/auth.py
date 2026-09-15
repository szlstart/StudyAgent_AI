"""Name + grade + semester local login (first login auto-registers)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.agents.memory import MemoryService
from src.api.dependencies import current_user
from src.domain.curriculum import profile_grade_label
from src.services.auth import COOKIE_NAME, SESSION_DAYS, CurrentUser, get_auth_store, user_data_dir
from src.services.evermemos import get_evermemos_service

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    grade: int = Field(ge=1, le=9)
    semester: str


class SchoolUpdate(BaseModel):
    grade: int = Field(ge=1, le=9)
    semester: str


async def _sync_profile_memory(user: CurrentUser) -> None:
    memory = MemoryService(user_data_dir(user.id))
    updated = await memory.update_preferences(
        nickname=user.name,
        grade=profile_grade_label(user.grade, user.semester),
    )
    async def log_long_term_profile() -> None:
        try:
            await asyncio.wait_for(
                get_evermemos_service(user.id).log_user_profile(updated.preferences.model_dump()),
                timeout=8,
            )
        except Exception:
            pass

    # EverMemOS 属于增强能力，不能阻塞本机登录。
    asyncio.create_task(log_long_term_profile())


@router.post("/login")
async def login(body: LoginRequest, response: Response):
    try:
        user, token, is_new = get_auth_store().login(body.name, body.grade, body.semester)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response.set_cookie(
        COOKIE_NAME, token, max_age=SESSION_DAYS * 86400, httponly=True,
        samesite="lax", secure=False, path="/",
    )
    await _sync_profile_memory(user)
    return {"ok": True, "is_new": is_new, "user": user.to_dict()}


@router.get("/me")
async def me(user: CurrentUser = Depends(current_user)):
    return {"user": user.to_dict()}


@router.patch("/profile")
async def update_school(body: SchoolUpdate, user: CurrentUser = Depends(current_user)):
    try:
        updated = get_auth_store().update_school(user.id, body.grade, body.semester)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _sync_profile_memory(updated)
    return {"ok": True, "user": updated.to_dict()}


@router.post("/logout")
async def logout(request: Request, response: Response):
    get_auth_store().logout(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}
