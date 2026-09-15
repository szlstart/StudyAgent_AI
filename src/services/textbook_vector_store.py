"""Qdrant-backed persistent vector storage for official textbook chunks."""

from __future__ import annotations

import os
import uuid
from typing import Any, Iterable

from qdrant_client import AsyncQdrantClient, QdrantClient, models


class TextbookVectorStore:
    def __init__(self) -> None:
        self.url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
        self.collection = os.getenv("QDRANT_TEXTBOOK_COLLECTION", "studybuddy_textbooks")

    @staticmethod
    def _filter(textbook_id: str, source_sha256: str | None = None) -> models.Filter:
        must: list[models.FieldCondition] = [
            models.FieldCondition(
                key="textbook_id", match=models.MatchValue(value=textbook_id)
            )
        ]
        if source_sha256:
            must.append(models.FieldCondition(
                key="source_sha256", match=models.MatchValue(value=source_sha256)
            ))
        return models.Filter(must=must)

    def has_textbook(self, textbook_id: str, source_sha256: str, expected: int) -> bool:
        """Verify that Qdrant contains every chunk referenced by the manifest."""
        client: QdrantClient | None = None
        try:
            client = QdrantClient(url=self.url, timeout=5)
            if not client.collection_exists(self.collection):
                return False
            count = client.count(
                collection_name=self.collection,
                count_filter=self._filter(textbook_id, source_sha256),
                exact=True,
            ).count
            return count >= expected > 0
        except Exception:
            return False
        finally:
            if client is not None:
                client.close()

    async def ensure_collection(self, dimensions: int) -> None:
        client = AsyncQdrantClient(url=self.url, timeout=30)
        try:
            if not await client.collection_exists(self.collection):
                await client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=dimensions, distance=models.Distance.COSINE
                    ),
                )
                await client.create_payload_index(
                    collection_name=self.collection,
                    field_name="textbook_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
                await client.create_payload_index(
                    collection_name=self.collection,
                    field_name="source_sha256",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
        finally:
            await client.close()

    async def replace_textbook(
        self,
        textbook_id: str,
        source_sha256: str,
        chunks: Iterable[Any],
    ) -> None:
        chunks = list(chunks)
        dimensions = len(chunks[0].embedding or []) if chunks else 0
        if not dimensions:
            raise RuntimeError("教材分块没有 embedding")
        await self.ensure_collection(dimensions)

        client = AsyncQdrantClient(url=self.url, timeout=60)
        try:
            points: list[models.PointStruct] = []
            for chunk in chunks:
                metadata = dict(chunk.metadata or {})
                point_key = (
                    f"{textbook_id}:{source_sha256}:"
                    f"{metadata.get('page', 0)}:{metadata.get('start_pos', 0)}"
                )
                payload = {
                    **metadata,
                    "textbook_id": textbook_id,
                    "source_sha256": source_sha256,
                    "content": chunk.content,
                }
                points.append(models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, point_key)),
                    vector=chunk.embedding,
                    payload=payload,
                ))

            for start in range(0, len(points), 128):
                await client.upsert(
                    collection_name=self.collection,
                    points=points[start:start + 128],
                    wait=True,
                )

            await client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(filter=models.Filter(
                    must=[models.FieldCondition(
                        key="textbook_id", match=models.MatchValue(value=textbook_id)
                    )],
                    must_not=[models.FieldCondition(
                        key="source_sha256", match=models.MatchValue(value=source_sha256)
                    )],
                )),
                wait=True,
            )
        finally:
            await client.close()

    async def search(
        self, textbook_id: str, source_sha256: str, vector: list[float], limit: int
    ) -> list[dict[str, Any]]:
        client = AsyncQdrantClient(url=self.url, timeout=30)
        try:
            response = await client.query_points(
                collection_name=self.collection,
                query=vector,
                query_filter=self._filter(textbook_id, source_sha256),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        finally:
            await client.close()
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            content = str(payload.pop("content", ""))
            payload.pop("source_sha256", None)
            results.append({"content": content, "metadata": payload, "score": point.score})
        return results
