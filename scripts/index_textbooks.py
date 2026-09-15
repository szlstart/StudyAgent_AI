#!/usr/bin/env python
"""Batch index official textbooks with resumable per-book status."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.explain.kb_manager import TextbookKBManager
from src.domain.curriculum import allowed_subjects


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grade", type=int, choices=range(1, 10))
    parser.add_argument("--semester", choices=("upper", "lower"))
    parser.add_argument("--subject")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manager = TextbookKBManager(ROOT)
    failures = []
    consecutive_failures = 0
    for grade in range(1, 10):
        if args.grade and grade != args.grade:
            continue
        for semester in ("upper", "lower"):
            if args.semester and semester != args.semester:
                continue
            for subject in allowed_subjects(grade):
                if args.subject and subject != args.subject:
                    continue
                status = manager.status(grade, semester, subject)
                print(f"[{status['textbook_id']}] {status['pdf_filename']}", flush=True)
                if status["indexed"] and not args.force:
                    print("  已存在，跳过", flush=True)
                    continue
                ok = await manager.reindex(grade, semester, subject)
                print("  完成" if ok else "  失败", flush=True)
                if not ok:
                    failures.append(status["textbook_id"])
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        print("连续 3 本失败，停止批处理以避免重复无效请求。", file=sys.stderr)
                        return 1
                else:
                    consecutive_failures = 0
    if failures:
        print("失败：" + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
