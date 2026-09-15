"""
EverMemOS API Client
====================

封装对本地 EverOS HTTP API 的底层调用。

API 参考：
  POST /api/v2/memory/add      — 写入消息
  POST /api/v2/memory/flush    — 提取并持久化记忆
  POST /api/v2/memory/search   — 混合检索记忆
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from src.logging import get_logger

logger = get_logger("EverMemOSClient")

_BASE_URL = os.getenv("EVERMEMOS_BASE_URL", "http://127.0.0.1:1995")
_API_KEY = os.getenv("EVERMEMOS_API_KEY", "")
_APP_ID = os.getenv("EVERMEMOS_APP_ID", "studybuddy")
_PROJECT_ID = os.getenv("EVERMEMOS_PROJECT_ID", "learning")
_RETRIEVE_METHOD = os.getenv("EVERMEMOS_RETRIEVE_METHOD", "hybrid")
_TIMEOUT = 120  # 记忆提取会调用 LLM，首次写入可能较慢


class EverMemOSClient:
    """
    EverMemOS REST API 的轻量异步客户端。

    所有方法失败时只记录 warning，不抛异常，
    保证调用方（后台任务）不会因此崩溃。
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
    ) -> None:
        self._api_key = api_key or _API_KEY
        self._base_url = (base_url or _BASE_URL).rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if self._api_key:
            self._headers["Authorization"] = f"Bearer {self._api_key}"

    # ─────────────────────────────────────────
    # Write — 写入一条消息
    # ─────────────────────────────────────────

    async def add_message(
        self,
        sender: str,
        content: str,
        *,
        role: str = "user",
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        sender_name: Optional[str] = None,
        flush: bool = True,
        message_id: Optional[str] = None,
        create_time: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        向 EverMemOS 写入一条消息，触发异步记忆提取。

        Args:
            sender:      消息发送者 ID（学生唯一标识）
            content:     消息正文
            role:        "user" | "assistant"
            group_id:    组织 / 群组 ID（留空则 API 自动生成）
            flush:       True = 立即触发记忆提取边界（本条消息为一个完整事件）

        Returns:
            {"status": "queued", "request_id": "..."}
        """
        del group_name, message_id  # EverOS v2 scopes memory with app/project/session IDs.

        created = create_time or datetime.now(timezone.utc).isoformat()
        try:
            timestamp_ms = int(
                datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                * 1000
            )
        except ValueError:
            timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        session_id = group_id or f"studybuddy-{sender}"
        payload: dict[str, Any] = {
            "session_id": session_id,
            "app_id": _APP_ID,
            "project_id": _PROJECT_ID,
            "messages": [
                {
                    "sender_id": sender,
                    "sender_name": sender_name,
                    "role": role,
                    "timestamp": timestamp_ms,
                    "content": content,
                }
            ],
        }

        result = await self._post("/api/v2/memory/add", payload)
        if result.get("status") == "error" or not flush:
            return result

        flush_result = await self._post(
            "/api/v2/memory/flush",
            {
                "session_id": session_id,
                "app_id": _APP_ID,
                "project_id": _PROJECT_ID,
            },
        )
        return flush_result

    # ─────────────────────────────────────────
    # Search — 语义检索记忆
    # ─────────────────────────────────────────

    async def search(
        self,
        query: str,
        user_id: str,
        *,
        memory_types: Optional[list[str]] = None,
        top_k: int = 5,
        retrieve_method: str = "hybrid",
        group_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        按语义查询相关记忆。

        Args:
            query:          查询文本
            user_id:        学生唯一 ID
            memory_types:   过滤类型，可选 profile / episodic_memory / event_log / foresight
            top_k:          返回条数
            retrieve_method: keyword | vector | hybrid | rrf | agentic

        Returns:
            {"status": "ok", "result": {"memories": [...], "profiles": [...], ...}}
        """
        del memory_types, group_ids  # EverOS v2 searches the scoped user memory.
        payload: dict[str, Any] = {
            "user_id": user_id,
            "app_id": _APP_ID,
            "project_id": _PROJECT_ID,
            "query": query,
            "method": retrieve_method or _RETRIEVE_METHOD,
            "top_k": top_k,
        }
        return await self._post("/api/v2/memory/search", payload)

    async def health(self) -> dict[str, Any]:
        """Return the local EverOS health/capability response."""
        return await self._get("/health", {})

    # ─────────────────────────────────────────
    # Internal HTTP helpers
    # ─────────────────────────────────────────

    async def _post(self, path: str, payload: dict) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(url, json=payload, headers=self._headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"[EverMemOS] POST {path} 失败: HTTP {e.response.status_code}"
            )
            return {"status": "error", "detail": str(e)}
        except Exception as e:
            logger.warning(f"[EverMemOS] POST {path} 网络错误: {e}")
            return {"status": "error", "detail": str(e)}

    async def _get(self, path: str, params: dict) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, params=params, headers=self._headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"[EverMemOS] GET {path} 失败: HTTP {e.response.status_code}"
            )
            return {"status": "error", "detail": str(e)}
        except Exception as e:
            logger.warning(f"[EverMemOS] GET {path} 网络错误: {e}")
            return {"status": "error", "detail": str(e)}
