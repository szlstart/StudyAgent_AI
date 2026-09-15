"""Grade/semester/subject-aware Qdrant textbook knowledge-base manager."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.domain.curriculum import (
    SUBJECTS, allowed_subjects, grade_label, semester_label,
    textbook_filename, textbook_id, textbook_path,
)
from src.logging import get_logger
from src.services.auth import get_auth_store
from src.services.textbook_vector_store import TextbookVectorStore

logger = get_logger("KBManager")
_locks: dict[str, asyncio.Lock] = {}
_hash_cache: dict[str, tuple[int, int, str]] = {}


def _lock_for(key: str) -> asyncio.Lock:
    if key not in _locks:
        _locks[key] = asyncio.Lock()
    return _locks[key]


def _sha256(path: Path) -> str:
    stat = path.stat()
    key = str(path)
    cached = _hash_cache.get(key)
    signature = (stat.st_mtime_ns, stat.st_size)
    if cached and cached[:2] == signature:
        return cached[2]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _hash_cache[key] = (signature[0], signature[1], value)
    return value


class TextbookKBManager:
    """Build and query one persistent vector index for every official textbook."""

    CHUNK_SIZE = 700
    CHUNK_OVERLAP = 100
    INDEX_VERSION = 4

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = project_root or Path(__file__).resolve().parent.parent.parent.parent
        self._textbook_root = self._project_root / "TextBook"
        self._index_root = self._project_root / "data" / "textbook_indexes"
        self._ocr_root = self._project_root / "data" / "textbook_ocr"
        self._vectors = TextbookVectorStore()
        self._index_root.mkdir(parents=True, exist_ok=True)
        self._ocr_root.mkdir(parents=True, exist_ok=True)

    def supports_subject(self, subject: str, grade: int | None = None) -> bool:
        return subject in SUBJECTS and (grade is None or subject in allowed_subjects(grade))

    def get_textbook_desc(self, grade: int, semester: str, subject: str) -> str:
        return textbook_filename(grade, semester, subject).removesuffix(".pdf")

    def source_path(self, grade: int, semester: str, subject: str) -> Path:
        return textbook_path(self._textbook_root, grade, semester, subject)

    def index_dir(self, grade: int, semester: str, subject: str) -> Path:
        return self._index_root / textbook_id(grade, semester, subject)

    def _manifest_path(self, grade: int, semester: str, subject: str) -> Path:
        return self.index_dir(grade, semester, subject) / "manifest.json"

    def ocr_path(self, grade: int, semester: str, subject: str) -> Path:
        return self._ocr_root / f"{textbook_id(grade, semester, subject)}.jsonl"

    def _load_ocr_pages(self, grade: int, semester: str, subject: str) -> dict[int, str]:
        path = self.ocr_path(grade, semester, subject)
        if not path.exists():
            return {}
        pages: dict[int, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                page = int(record["page"])
                text = str(record.get("text", "")).strip()
                if text:
                    pages[page] = text
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
        return pages

    def get_manifest(self, grade: int, semester: str, subject: str) -> dict[str, Any]:
        path = self._manifest_path(grade, semester, subject)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def is_indexed(self, grade: int, semester: str, subject: str) -> bool:
        manifest = self.get_manifest(grade, semester, subject)
        source = self.source_path(grade, semester, subject)
        if not source.exists() or not manifest:
            return False
        try:
            from src.services.embedding.config import get_embedding_config
            model = get_embedding_config().model
        except Exception:
            model = os.getenv("EMBEDDING_MODEL", "")
        valid_manifest = (
            manifest.get("index_version") == self.INDEX_VERSION
            and manifest.get("storage") == "qdrant"
            and manifest.get("source_sha256") == _sha256(source)
            and manifest.get("embedding_model") == model
        )
        if not valid_manifest:
            return False
        return self._vectors.has_textbook(
            textbook_id(grade, semester, subject),
            manifest["source_sha256"],
            int(manifest.get("chunk_count", 0)),
        )

    def status(self, grade: int, semester: str, subject: str) -> dict[str, Any]:
        source = self.source_path(grade, semester, subject)
        tid = textbook_id(grade, semester, subject)
        manifest = self.get_manifest(grade, semester, subject)
        db_status = get_auth_store().get_textbook_status(tid) or {}
        indexed = self.is_indexed(grade, semester, subject)
        state = "ready" if indexed else db_status.get("status", "not_indexed")
        if state == "ready" and not indexed:
            state = "stale"
        return {
            "textbook_id": tid, "subject": subject,
            "label": SUBJECTS[subject]["label"], "icon": SUBJECTS[subject]["icon"],
            "grade": grade, "grade_label": grade_label(grade),
            "semester": semester, "semester_label": semester_label(semester),
            "pdf_filename": textbook_filename(grade, semester, subject),
            "pdf_exists": source.exists(),
            "file_size_kb": round(source.stat().st_size / 1024) if source.exists() else 0,
            "indexed": indexed, "status": state,
            "page_count": manifest.get("page_count", db_status.get("page_count", 0)),
            "chunk_count": manifest.get("chunk_count", db_status.get("chunk_count", 0)),
            "updated_at": manifest.get("updated_at", db_status.get("updated_at", "")),
            "error": "索引失败，请重试或查看服务日志" if state == "failed" else "",
        }

    async def ensure_indexed(self, grade: int, semester: str, subject: str) -> str | None:
        tid = textbook_id(grade, semester, subject)
        source = self.source_path(grade, semester, subject)
        if self.is_indexed(grade, semester, subject):
            return tid
        if not source.exists():
            logger.warning(f"教材不存在: {source}")
            return None
        async with _lock_for(tid):
            if self.is_indexed(grade, semester, subject):
                return tid
            ok = await self._build_index(grade, semester, subject)
        return tid if ok else None

    async def reindex(self, grade: int, semester: str, subject: str) -> bool:
        tid = textbook_id(grade, semester, subject)
        async with _lock_for(tid):
            return await self._build_index(grade, semester, subject)

    async def search_sources(
        self, grade: int, semester: str, subject: str, query: str, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        # Student requests should not unexpectedly launch a full PDF embedding job.
        # Reindexing remains an explicit action on the knowledge-base page.
        if not await asyncio.to_thread(self.is_indexed, grade, semester, subject):
            return []
        if not query.strip():
            return []
        try:
            tid = textbook_id(grade, semester, subject)
            from src.services.embedding import get_embedding_client
            manifest = self.get_manifest(grade, semester, subject)
            vectors = await get_embedding_client().embed([query[:2_000]])
            if not vectors:
                return []
            results = await self._vectors.search(
                tid, manifest["source_sha256"], vectors[0], min(max(top_k, 1), 10)
            )
            min_score = float(os.getenv("RAG_MIN_SCORE", "0.35"))
            return [
                item for item in results
                if item.get("content", "").strip()
                and float(item.get("score") or 0) >= min_score
            ]
        except Exception as exc:
            # 教材检索是增强能力，不应因向量库或 embedding 短时故障让
            # 整个答疑/批改变成 500。调用方会明确标记为未使用教材依据。
            logger.warning(f"教材检索暂时不可用，已降级为模型直答: {exc}")
            return []

    async def search(
        self, grade: int, semester: str, subject: str, query: str, top_k: int = 5,
    ) -> list[str]:
        sources = await self.search_sources(grade, semester, subject, query, top_k)
        output = []
        for item in sources:
            page = item.get("metadata", {}).get("page")
            prefix = f"[第 {page} 页] " if page else ""
            output.append(prefix + item["content"])
        return output

    async def _build_index(self, grade: int, semester: str, subject: str) -> bool:
        from datetime import datetime, timezone
        from src.services.embedding.config import get_embedding_config
        from src.services.rag.components.embedders.openai import OpenAIEmbedder
        from src.services.rag.types import Chunk, Document

        tid = textbook_id(grade, semester, subject)
        source = self.source_path(grade, semester, subject)
        filename = source.name
        cfg = get_embedding_config()
        source_hash = ""
        store = get_auth_store()
        try:
            if not source.exists():
                raise RuntimeError("教材 PDF 不存在")
            source_hash = _sha256(source)
            store.set_textbook_status(tid, filename, "indexing", source_sha256=source_hash, embedding_model=cfg.model)
            try:
                import pymupdf as fitz
            except ImportError as exc:
                raise RuntimeError("缺少 PyMuPDF，无法提取教材 PDF") from exc
            pdf = fitz.open(source)
            chunks: list[Chunk] = []
            text_pages = 0
            ocr_pages = self._load_ocr_pages(grade, semester, subject)
            ocr_pages_used = 0
            step = self.CHUNK_SIZE - self.CHUNK_OVERLAP
            for page_no, page in enumerate(pdf, start=1):
                text_value = page.get_text("text").strip()
                if len(text_value) < 30 and ocr_pages.get(page_no):
                    text_value = ocr_pages[page_no]
                    ocr_pages_used += 1
                if not text_value:
                    continue
                text_pages += 1
                for start in range(0, len(text_value), step):
                    content = text_value[start:start + self.CHUNK_SIZE].strip()
                    if content:
                        chunks.append(Chunk(content=content, metadata={
                            "textbook_id": tid, "filename": filename, "grade": grade,
                            "semester": semester, "subject": subject, "page": page_no,
                            "start_pos": start,
                        }))
            page_count = len(pdf)
            pdf.close()
            if not chunks:
                raise RuntimeError("PDF 没有可提取文字；该教材可能是扫描版，需要教材 OCR")
            if text_pages / max(page_count, 1) < 0.5:
                raise RuntimeError(f"仅 {text_pages}/{page_count} 页可提取文字，需要教材 OCR")

            doc = Document(content="", file_path=str(source), metadata={"filename": filename}, chunks=chunks)
            # DashScope OpenAI-compatible embeddings caps a request at 20 inputs.
            doc = await OpenAIEmbedder(batch_size=20).process(doc)
            await self._vectors.replace_textbook(tid, source_hash, doc.chunks)

            now = datetime.now(timezone.utc).isoformat()
            manifest = {
                "index_version": self.INDEX_VERSION, "storage": "qdrant",
                "collection": self._vectors.collection,
                "textbook_id": tid, "filename": filename,
                "source_sha256": source_hash, "embedding_model": cfg.model,
                "embedding_dimension": len(chunks[0].embedding or []),
                "page_count": page_count, "text_page_count": text_pages,
                "ocr_page_count": ocr_pages_used,
                "chunk_count": len(chunks), "chunk_size": self.CHUNK_SIZE,
                "chunk_overlap": self.CHUNK_OVERLAP, "updated_at": now,
            }
            index_dir = self.index_dir(grade, semester, subject)
            index_dir.mkdir(parents=True, exist_ok=True)
            (index_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            store.set_textbook_status(
                tid, filename, "ready", source_sha256=source_hash, embedding_model=cfg.model,
                page_count=page_count, chunk_count=len(chunks),
            )
            logger.info(f"教材索引完成: {tid}, pages={page_count}, chunks={len(chunks)}")
            return True
        except Exception as exc:
            store.set_textbook_status(
                tid, filename, "failed", source_sha256=source_hash,
                embedding_model=cfg.model, error=str(exc),
            )
            logger.error(f"教材索引失败 {tid}: {exc}", exc_info=True)
            return False
