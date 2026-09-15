from __future__ import annotations

import io
import asyncio

import pytest
import pymupdf
from PIL import Image
from pillow_heif import register_heif_opener
from starlette.datastructures import Headers, UploadFile

from src.api.routers.explain import extract_from_file
from src.services.auth import CurrentUser
from src.services.image_uploads import normalize_uploaded_image


def _image_bytes(fmt: str = "PNG") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), (30, 90, 140)).save(output, format=fmt)
    return output.getvalue()


def test_heic_phone_photo_is_converted_to_jpeg():
    register_heif_opener()
    source = io.BytesIO()
    Image.new("RGB", (24, 16), (25, 80, 140)).save(source, format="HEIF")

    converted, mime = normalize_uploaded_image(
        source.getvalue(), "image/heic", "作业照片.HEIC",
    )

    result = Image.open(io.BytesIO(converted))
    assert mime == "image/jpeg"
    assert result.format == "JPEG"
    assert result.size == (24, 16)


def test_regular_supported_image_is_not_reencoded():
    raw = _image_bytes("PNG")
    converted, mime = normalize_uploaded_image(raw, "image/png", "homework.png")
    assert converted == raw
    assert mime == "image/png"


def test_supported_suffix_recovers_missing_browser_mime():
    raw = _image_bytes("JPEG")
    converted, mime = normalize_uploaded_image(raw, "", "camera-photo.JPG")
    assert converted == raw
    assert mime == "image/jpeg"


def test_fake_image_payload_is_rejected_even_with_image_mime():
    with pytest.raises(ValueError, match="损坏|不符"):
        normalize_uploaded_image(b"not-an-image", "image/png", "fake.png")


def test_unsupported_upload_has_actionable_error():
    with pytest.raises(ValueError, match="不支持的图片格式"):
        normalize_uploaded_image(b"content", "application/pdf", "homework.pdf")


def test_image_over_20mb_is_rejected():
    with pytest.raises(ValueError, match="20 MB"):
        normalize_uploaded_image(
            b"x" * (20 * 1024 * 1024 + 1), "image/jpeg", "too-large.jpg",
        )


def test_text_pdf_is_extracted_for_question_inquiry():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Quadratic equation example")
    raw = document.tobytes()
    document.close()
    upload = UploadFile(
        io.BytesIO(raw), filename="question.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )
    user = CurrentUser(
        id="test-user", name="测试", grade=9, semester="lower",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )

    result = asyncio.run(extract_from_file(upload, "math", user))

    assert result["source"] == "pdf"
    assert result["page_count"] == 1
    assert result["truncated"] is False
    assert "Quadratic equation example" in result["text"]
