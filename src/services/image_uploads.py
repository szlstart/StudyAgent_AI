"""Normalize user-uploaded images before passing them to vision models."""

from __future__ import annotations

import io
import warnings
from pathlib import Path


SUPPORTED_IMAGE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/heic",
    "image/heif",
}
HEIF_SUFFIXES = {".heic", ".heif"}
MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif",
    ".heic": "image/heic", ".heif": "image/heif",
}
MIME_BY_PIL_FORMAT = {
    "JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif",
}
MAX_IMAGE_PIXELS = 40_000_000


def normalize_uploaded_image(
    content: bytes,
    content_type: str | None,
    filename: str | None,
    *,
    max_bytes: int = 20 * 1024 * 1024,
) -> tuple[bytes, str]:
    """Validate an uploaded image and convert HEIC/HEIF to JPEG.

    OpenAI-compatible vision endpoints do not consistently accept Apple's
    HEIC container.  Phone photos in that format are decoded locally and
    converted to an EXIF-corrected RGB JPEG.  Other supported formats remain
    byte-for-byte unchanged.
    """
    if not content:
        raise ValueError("图片为空")
    if len(content) > max_bytes:
        raise ValueError("图片不能超过 20 MB")

    mime = (content_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(filename or "").suffix.lower()
    if not mime or mime == "application/octet-stream":
        mime = MIME_BY_SUFFIX.get(suffix, mime)
    is_heif = mime in {"image/heic", "image/heif"} or suffix in HEIF_SUFFIXES

    if is_heif:
        try:
            from PIL import Image, ImageOps
            from pillow_heif import register_heif_opener

            register_heif_opener()
            with Image.open(io.BytesIO(content)) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=92, optimize=True)
            return output.getvalue(), "image/jpeg"
        except Exception as exc:
            raise ValueError("HEIC/HEIF 图片解析失败，请改用 JPG 或 PNG 后重试") from exc

    if mime not in SUPPORTED_IMAGE_MIMES:
        raise ValueError("不支持的图片格式，请使用 JPG、PNG、WEBP、GIF、HEIC 或 HEIF")

    # MIME 和扩展名都由客户端提供，不能据此相信文件确实是图片。先让
    # Pillow 解码文件头并验证容器，同时拒绝会造成内存耗尽的超大像素图。
    try:
        from PIL import Image

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as opened:
                actual_format = (opened.format or "").upper()
                width, height = opened.size
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("图片分辨率过大，请压缩后重试")
                opened.verify()
        actual_mime = MIME_BY_PIL_FORMAT.get(actual_format)
        if actual_mime is None:
            raise ValueError("图片内容格式不受支持，请改用 JPG、PNG、WEBP 或 GIF")
        return content, actual_mime
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("图片文件已损坏或内容与图片格式不符，请重新选择") from exc
