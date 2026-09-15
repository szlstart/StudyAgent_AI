#!/usr/bin/env python3
"""Create resumable local OCR caches for scanned textbooks on macOS."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
TEXTBOOK_ROOT = ROOT / "TextBook"
OCR_ROOT = ROOT / "data" / "textbook_ocr"
SWIFT_SOURCE = ROOT / "scripts" / "macos_vision_pdf_ocr.swift"
OCR_BINARY = ROOT / "data" / "tools" / "macos_vision_pdf_ocr"


def needs_ocr(pdf_path: Path) -> tuple[bool, int, int]:
    document = pymupdf.open(pdf_path)
    page_count = len(document)
    text_pages = sum(1 for page in document if len(page.get_text("text").strip()) >= 30)
    document.close()
    return text_pages / max(page_count, 1) < 0.5, page_count, text_pages


def compile_helper() -> None:
    OCR_BINARY.parent.mkdir(parents=True, exist_ok=True)
    if OCR_BINARY.exists() and OCR_BINARY.stat().st_mtime >= SWIFT_SOURCE.stat().st_mtime:
        return
    print("正在编译 macOS Vision OCR 辅助程序…", flush=True)
    subprocess.run(["swiftc", str(SWIFT_SOURCE), "-o", str(OCR_BINARY)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=None, help="调试时限制每本页数")
    parser.add_argument("--match", default="", help="只处理路径中包含此文字的教材")
    args = parser.parse_args()

    candidates: list[tuple[Path, int]] = []
    for pdf_path in sorted(TEXTBOOK_ROOT.glob("*/*.pdf")):
        if args.match and args.match not in str(pdf_path):
            continue
        needed, page_count, text_pages = needs_ocr(pdf_path)
        if needed:
            candidates.append((pdf_path, page_count))
            print(f"扫描版教材: {pdf_path.name} ({text_pages}/{page_count} 页含文本)", flush=True)

    if not candidates:
        print("没有需要 OCR 的教材。")
        return 0

    compile_helper()
    OCR_ROOT.mkdir(parents=True, exist_ok=True)
    for index, (pdf_path, page_count) in enumerate(candidates, start=1):
        grade = pdf_path.parent.name
        stem_parts = pdf_path.stem.split("_")
        subject_label, _, semester_label = stem_parts
        subject_keys = {
            "语文": "chinese", "数学": "math", "英语": "english", "物理": "physics",
            "生物": "biology", "历史": "history", "化学": "chemistry", "地理": "geography",
        }
        grade_numbers = {
            "一年级": 1, "二年级": 2, "三年级": 3, "四年级": 4, "五年级": 5,
            "六年级": 6, "七年级": 7, "八年级": 8, "九年级": 9,
        }
        semester = "upper" if semester_label == "上册" else "lower"
        textbook_id = f"grade_{grade_numbers[grade]:02d}_{semester}_{subject_keys[subject_label]}"
        output = OCR_ROOT / f"{textbook_id}.jsonl"
        command = [str(OCR_BINARY), str(pdf_path), str(output)]
        if args.max_pages:
            command.append(str(args.max_pages))
        print(f"[{index}/{len(candidates)}] OCR {pdf_path.name}，共 {page_count} 页（支持断点续跑）", flush=True)
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
