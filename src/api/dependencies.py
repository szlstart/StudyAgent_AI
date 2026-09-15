"""FastAPI request dependencies."""

from fastapi import HTTPException, Request

from src.services.auth import CurrentUser


def current_user(request: Request) -> CurrentUser:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return user
