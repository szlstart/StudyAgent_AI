"""
Homework Grading API Router
============================

Endpoints
---------
POST /api/v1/homework/grade    — Upload image(s) + subject, run full 4-step pipeline
GET  /api/v1/homework/health   — Service health check

Pipeline
--------
图片上传
  ↓ OCRAgent        — 识别题目 + 学生作答（多图并行）→ list[ExtractedQuestion]
  ↓ GradeAgent      — 逐题批改                      → list[GradedQuestion]
  ↓ KnowPointAgent  — 知识点 + 难度标注             → list[GradedQuestion]（已注解）
  ↓ ExamTagAgent    — 试卷特征 + 薄弱知识点          → (exam_tags, weak_points)
  ↓ HomeworkResult  — 汇总、保存错题与历史后返回
"""

from __future__ import annotations

import asyncio
import base64
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Project root in path ──────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.homework import (
    ExamTagAgent,
    GradeAgent,
    HomeworkResult,
    KnowPointAgent,
    build_homework_ocr_agent,
)
from src.agents.homework.models import GradedQuestion, GradeResult, QuestionType
from src.agents.homework.ocr_agent import OCRSubmissionError
from src.agents.homework.wrong_book_service import WrongBookService
from src.services.llm.exceptions import LLMError
from src.agents.memory import MemoryService, PerformanceRecord
from src.agents.explain.kb_manager import TextbookKBManager
from src.api.dependencies import current_user
from src.domain.curriculum import allowed_subjects, textbook_id
from src.logging import get_logger
from src.services.auth import CurrentUser, user_data_dir
from src.services.evermemos import get_evermemos_service
from src.services.image_uploads import normalize_uploaded_image
from src.services.question_bank import QuestionBankService
from src.services.atomic_io import atomic_write_bytes, atomic_write_text
from src.services.user_files import resolve_user_file
from src.services.homework_images import attach_question_crops, create_question_crops

# ── Logger ────────────────────────────────────────────────────────────────────
logger = get_logger("Homework")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_HOMEWORK_PAGES = 12
_PROGRESS_TTL_SECONDS = 15 * 60

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter()

_mem_services: dict[str, MemoryService] = {}
_grade_progress: dict[tuple[str, str], dict] = {}


def _textbook_reference(snippet: str) -> str:
    """Extract a truthful chapter/page label from one retrieved textbook chunk."""
    page_match = re.search(r"\[第\s*(\d+)\s*页\]", snippet)
    chapter_match = re.search(
        r"(第\s*[一二三四五六七八九十百零〇两\d]+\s*章[^\n。；]{0,36})",
        snippet,
    )
    parts: list[str] = []
    if chapter_match:
        chapter = re.sub(r"\s+", " ", chapter_match.group(1)).strip(" ：:")
        parts.append(chapter)
    if page_match:
        parts.append(f"第 {page_match.group(1)} 页")
    return " · ".join(parts)


def _attach_textbook_references(
    questions: list[GradedQuestion], snippets: list[str],
) -> None:
    """Rank actual RAG chunks for every question and attach up to two citations."""
    candidates: list[tuple[str, str]] = []
    for snippet in snippets:
        reference = _textbook_reference(snippet)
        if reference and all(reference != existing[0] for existing in candidates):
            candidates.append((reference, snippet))
    for question in questions:
        query = "".join(
            ch for ch in f"{question.question_text}{' '.join(question.knowledge_points)}"
            if "\u4e00" <= ch <= "\u9fff" or ch.isalnum()
        ).casefold()
        grams = {query[index:index + 2] for index in range(max(0, len(query) - 1))}
        ranked: list[tuple[int, int, str]] = []
        for index, (reference, snippet) in enumerate(candidates):
            normalized = "".join(ch for ch in snippet if "\u4e00" <= ch <= "\u9fff" or ch.isalnum()).casefold()
            score = sum(1 for gram in grams if gram in normalized)
            ranked.append((score, -index, reference))
        ranked.sort(reverse=True)
        question.textbook_refs = [item[2] for item in ranked[:2] if item[0] > 0]
        if not question.textbook_refs and candidates:
            question.textbook_refs = [candidates[0][0]]


def _valid_request_id(request_id: str) -> bool:
    return isinstance(request_id, str) and bool(
        re.fullmatch(r"[A-Za-z0-9_-]{8,80}", request_id)
    )


def _set_grade_progress(
    user_id: str,
    request_id: str,
    stage: str,
    message: str,
    percent: int,
) -> None:
    if not _valid_request_id(request_id):
        return
    now = time.monotonic()
    expired = [
        key for key, value in _grade_progress.items()
        if now - float(value.get("updated_monotonic", now)) > _PROGRESS_TTL_SECONDS
    ]
    for key in expired:
        _grade_progress.pop(key, None)
    _grade_progress[(user_id, request_id)] = {
        "request_id": request_id,
        "stage": stage,
        "message": message,
        "percent": max(0, min(int(percent), 100)),
        "updated_monotonic": now,
    }

_ERR_LABEL_CN: dict[str, str] = {
    "calculation_error": "计算失误",
    "concept_confusion": "概念混淆",
    "reading_mistake":   "审题失误",
    "formula_wrong":     "公式错误",
    "sign_error":        "符号错误",
    "unit_error":        "单位错误",
    "incomplete":        "解答不完整",
    "logic_error":       "逻辑错误",
    "spelling_grammar":  "拼写/语法错误",
    "other":             "其他",
}


async def _build_grade_memory_hint(
    user_id: str, subject: str, query: str = "",
) -> tuple[str, list[str]]:
    """
    从记忆服务构造注入 GradeAgent 的 prompt 片段，同时返回薄弱知识点列表。

    Returns:
        (memory_hint_str, known_weak_points)
        - memory_hint_str  : 直接拼接到 GradeAgent system prompt 末尾
        - known_weak_points: 传给 KnowPointAgent 用于 is_weak_area 标注
    """
    try:
        mem_svc = _get_mem_service(user_id)
        memory = await mem_svc.load_memory()

        parts: list[str] = []

        # 高频错误类型（Top 3）
        err_counts = memory.error_pattern_counts.get(subject, {})
        if err_counts:
            top_errors = sorted(err_counts, key=lambda k: err_counts[k], reverse=True)[:3]
            errs_cn = "、".join(_ERR_LABEL_CN.get(e, e) for e in top_errors)
            parts.append(f"该学生在本科目的高频错误：{errs_cn}；仅供参考，当前错误原因必须根据本次实际作答自由判断")

        # 近期薄弱知识点（最近 6 个）
        weak_points = memory.knowledge_gaps.get(subject, [])[-6:]
        if weak_points:
            pts = "、".join(weak_points)
            parts.append(
                f"近期薄弱知识点：{pts}；"
                f"若该题涉及上述知识点，brief_comment 中可点出「注意这是你的薄弱环节」"
            )

        hint = ""
        if parts:
            hint = (
                "\n\n【学生历史档案（仅供参考，不影响判题标准）】\n"
                + "\n".join(f"• {p}" for p in parts)
            )

        # 正式判分只使用当前本地结构化画像；EverOS 中的历史自然语言记忆
        # 可能包含已撤销结论，因此只用于讲解个性化，绝不进入判分依据。
        return hint, weak_points

    except Exception as e:
        logger.warning(f"[Homework] 记忆档案加载失败（不影响批改）: {e}")
        return "", []

def _get_mem_service(user_id: str) -> MemoryService:
    if user_id not in _mem_services:
        _mem_services[user_id] = MemoryService(user_data_dir(user_id))
    return _mem_services[user_id]

# ── Question bank singleton ────────────────────────────────────────────────────
_question_banks: dict[str, QuestionBankService] = {}

def _get_question_bank(curriculum_id: str) -> QuestionBankService:
    if curriculum_id not in _question_banks:
        _question_banks[curriculum_id] = QuestionBankService(
            _project_root / "data" / "question_banks" / curriculum_id
        )
    return _question_banks[curriculum_id]


_kb_manager = TextbookKBManager(_project_root)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check(user: CurrentUser = Depends(current_user)):
    """作业批改模块健康检查。"""
    return {"status": "ok", "module": "homework"}


@router.get("/images/{filename}")
async def homework_image(
    filename: str,
    user: CurrentUser = Depends(current_user),
):
    """Serve only the signed-in student's saved homework image/crop."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,180}", filename) or filename.startswith("."):
        raise HTTPException(status_code=404, detail="题目图片不存在")
    path = resolve_user_file(
        user_data_dir(user.id) / "homework_images" / filename,
        user_data_dir(user.id),
    )
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="题目图片不存在")
    return FileResponse(
        path,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/progress/{request_id}")
async def grade_progress(
    request_id: str,
    user: CurrentUser = Depends(current_user),
):
    """Return real pipeline progress for the current student's request."""
    if not _valid_request_id(request_id):
        raise HTTPException(status_code=400, detail="批改任务 ID 格式不正确")
    progress = _grade_progress.get((user.id, request_id))
    if progress is None:
        return {
            "request_id": request_id,
            "stage": "queued",
            "message": "正在接收作业图片…",
            "percent": 2,
        }
    return {key: value for key, value in progress.items() if key != "updated_monotonic"}


@router.get("/question-bank/stats")
async def question_bank_stats(
    subject: str = "math",
    user: CurrentUser = Depends(current_user),
):
    """题库统计：各科目缓存的题目数量和命中次数。"""
    if subject != "all" and subject not in allowed_subjects(user.grade):
        raise HTTPException(status_code=400, detail="该科目不属于当前年级")
    curriculum_id = textbook_id(user.grade, user.semester, subject if subject != "all" else allowed_subjects(user.grade)[0])
    bank = _get_question_bank(curriculum_id)
    if subject == "all":
        return [
            _get_question_bank(textbook_id(user.grade, user.semester, s)).stats(s)
            for s in allowed_subjects(user.grade)
        ]
    return bank.stats(subject)


@router.post("/grade", response_model=HomeworkResult)
async def grade_homework(
    files: list[UploadFile] = File(
        ...,
        description="试卷图片（JPG / PNG / WEBP / GIF / HEIC / HEIF），支持多张；按图片内印刷题号整理顺序",
    ),
    subject: str = Form(
        default="math",
        description="科目由登录年级决定：chinese | math | english | physics | biology | history | chemistry | geography",
    ),
    record_type: str = Form(
        default="homework",
        description="类型：homework（作业）| exam（考试）",
    ),
    exam_name: str = Form(
        default="",
        description="考试名称（record_type=exam 时填写，如「期中考试」）",
    ),
    model_key: str = Form(
        default="",
        description="兼容旧前端；批改固定使用系统配置的 GPT-5.5",
    ),
    request_id: str = Form(
        default="",
        description="前端生成的批改任务 ID，用于查询真实处理进度",
    ),
    user: CurrentUser = Depends(current_user),
):
    """
    作业批改完整流程（同步，等待全部结果后返回）。

    接收一张或多张试卷图片，经过四个 AI 阶段处理后返回完整结构化结果：

    1. **OCR**        — 识别所有题目及学生作答
    2. **Grade**      — 逐题批改，给出正确答案和点评
    3. **KnowPoint**  — 标注每题知识点和难度等级
    4. **ExamTag**    — 分析整份试卷特征 + 找出薄弱知识点

    **Request (multipart/form-data)**
    - `files`   : 图片文件，至少 1 张
    - `subject` : 科目字符串（默认 `math`）

    **Response**
    - `HomeworkResult` JSON：含题目明细、批改统计、试卷标签、薄弱知识点
    """
    # ── 0. 验证科目 ─────────────────────────────────────────────────────────
    subject_key = subject.strip().lower()
    if subject_key not in allowed_subjects(user.grade):
        raise HTTPException(status_code=400, detail="该科目不属于当前年级")
    curriculum_id = textbook_id(user.grade, user.semester, subject_key)
    _set_grade_progress(user.id, request_id, "validating", "正在校验上传内容…", 4)
    if len(files) > MAX_HOMEWORK_PAGES:
        raise HTTPException(
            status_code=400,
            detail=f"一次最多上传 {MAX_HOMEWORK_PAGES} 页，请分批批改",
        )

    # ── 1. 读取文件，转 base64 ───────────────────────────────────────────────
    images: list[str] = []
    image_bytes: list[tuple[bytes, str, bytes, str]] = []
    for upload in files:
        original = await upload.read(MAX_UPLOAD_BYTES + 1)
        if not original:
            continue
        original_suffix = Path(upload.filename or "").suffix.lower()
        try:
            content, mime = normalize_uploaded_image(
                original, upload.content_type, upload.filename,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        b64 = base64.b64encode(content).decode()
        images.append(f"data:{mime};base64,{b64}")
        image_bytes.append((content, mime, original, original_suffix))

    if not images:
        raise HTTPException(status_code=400, detail="请上传至少一张试卷图片")

    logger.info(f"[Homework] 收到 {len(images)} 张图片，科目={subject_key}")
    _set_grade_progress(
        user.id, request_id, "received",
        f"已接收 {len(images)} 页图片，正在安全保存原图…", 10,
    )

    # 保存每一页原图，错题解析时按 page_index 回看对应页面。
    saved_image_paths: list[str] = []
    batch_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    try:
        img_dir = user_data_dir(user.id) / "homework_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        suffixes = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}
        for page_index, (raw, mime, original, original_suffix) in enumerate(image_bytes):
            img_file = img_dir / f"{batch_id}_{subject_key}_p{page_index + 1}{suffixes[mime]}"
            atomic_write_bytes(img_file, raw)
            saved_image_paths.append(str(img_file))
            # HEIC/HEIF 需要转换后才能稳定交给视觉模型；同时保留用户上传的
            # 原始文件，且只存放在该用户私有目录中。
            if original != raw and original_suffix in {".heic", ".heif"}:
                original_file = (
                    img_dir
                    / f"{batch_id}_{subject_key}_p{page_index + 1}_original{original_suffix}"
                )
                atomic_write_bytes(original_file, original)
    except Exception as img_exc:
        logger.warning(f"[Homework] 图片保存失败（不影响批改）: {img_exc}")

    # 全链路固定使用 .env 配置的 GPT-5.5，不接受前端切换模型。
    model_kw: dict = {}

    try:
        # ── 2. OCR ─────────────────────────────────────────────────────────
        logger.info("[Homework] 第一步：视觉 OCR 逐页识别中...")
        _set_grade_progress(
            user.id, request_id, "recognizing",
            f"GPT-5.5 正在读取 {len(images)} 页中的题号、题干、手写步骤和图形…", 18,
        )
        ocr_agent = build_homework_ocr_agent(None)
        try:
            questions = await ocr_agent.process(
                images=images,
                subject=subject_key,
                progress_callback=lambda processed, total, page_index, succeeded: _set_grade_progress(
                    user.id,
                    request_id,
                    "recognizing",
                    (
                        f"视觉识别中：已处理 {processed}/{total} 页（上传位置第 {page_index + 1} 页识别完成）…"
                        if succeeded else
                        f"视觉识别中：已处理 {processed}/{total} 页（上传位置第 {page_index + 1} 页识别失败，正在检查其余页）…"
                    ),
                    18 + int(20 * processed / max(total, 1)),
                ),
            )
        except OCRSubmissionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if not questions:
            raise HTTPException(
                status_code=422,
                detail="未能从图片中识别出任何题目，请检查图片质量或方向",
            )
        logger.info(f"[Homework] OCR 完成，共识别 {len(questions)} 道题")
        _set_grade_progress(
            user.id, request_id, "ordering",
            f"已识别 {len(questions)} 个小问，已按图片中的印刷题号排序…", 42,
        )

        # 只检索当前登录学生的年级、学期和所选科目教材。
        retrieval_query = "\n".join(
            f"{q.question_text} {q.visual_context}" for q in questions
        )[:1200]
        # ── 2.5 加载本地画像 + EverOS 相关记忆 ───────────────────────
        memory_hint, known_weak_points = await _build_grade_memory_hint(
            user.id, subject_key, retrieval_query,
        )

        _set_grade_progress(
            user.id, request_id, "retrieving",
            "正在检索当前年级、学期和科目的教材依据…", 49,
        )

        textbook_snippets = await _kb_manager.search(
            user.grade, user.semester, subject_key, retrieval_query, top_k=6,
        )
        textbook_context = "\n\n---\n\n".join(textbook_snippets)

        # ── 3. Grade（题库缓存优先，命中则跳过 LLM）──────────────────────
        logger.info("[Homework] 第二步：查题库 + 批改中...")
        _set_grade_progress(
            user.id, request_id, "grading",
            f"GPT-5.5 正在逐题核对学生答案并生成解题思路（共 {len(questions)} 小问）…", 60,
        )
        bank = _get_question_bank(curriculum_id)

        # 对每道题查题库，客观题命中则直接构造 GradedQuestion
        cached_map: dict[int, GradedQuestion] = {}
        miss_questions = []
        miss_indices: list[int] = []

        for i, q in enumerate(questions):
            qtype = q.question_type.value if hasattr(q.question_type, "value") else str(q.question_type)
            hit = await bank.lookup(q.question_text, q.student_answer, qtype, subject_key)
            if hit:
                grade_val = hit["grade"]
                try:
                    grade_enum = GradeResult(grade_val)
                except ValueError:
                    grade_enum = GradeResult.SKIP

                # 推算 earned_score
                earned = None
                if q.score_value is not None:
                    if grade_enum == GradeResult.CORRECT:
                        earned = q.score_value
                    elif grade_enum in (GradeResult.WRONG, GradeResult.BLANK):
                        earned = 0.0

                cached_map[i] = GradedQuestion(
                    number=q.number,
                    major_number=q.major_number,
                    sub_number=q.sub_number,
                    stem_text=q.stem_text,
                    subquestion_text=q.subquestion_text,
                    question_text=q.question_text,
                    student_answer=q.student_answer,
                    question_type=q.question_type,
                    score_value=q.score_value,
                    page_index=q.page_index,
                    visual_context=q.visual_context,
                    bounding_box=q.bounding_box,
                    correct_answer=hit["correct_answer"],
                    grade=grade_enum,
                    earned_score=earned,
                    error_type=hit.get("error_type"),
                    brief_comment=hit["brief_comment"],
                    knowledge_points=[],
                    difficulty=None,
                )
            else:
                miss_questions.append(q)
                miss_indices.append(i)

        # 只对题库未命中的题调用 LLM
        newly_graded: list[GradedQuestion] = []
        if miss_questions:
            logger.info("[Homework] Grade 模型：系统 GPT-5.5（视觉 + 教材 RAG）")
            grade_agent = GradeAgent()
            newly_graded = await grade_agent.process(
                miss_questions, subject=subject_key, memory_hint=memory_hint,
                images=images, textbook_context=textbook_context,
                progress_callback=lambda completed, total: _set_grade_progress(
                    user.id,
                    request_id,
                    "grading",
                    f"GPT-5.5 已完成 {completed}/{total} 个小问的判分与解题思路…",
                    60 + int(16 * completed / max(total, 1)),
                ),
            )

        # 合并结果，保持原始顺序
        graded_slots: list[GradedQuestion | None] = [None] * len(questions)
        for idx, gq in cached_map.items():
            graded_slots[idx] = gq
        for list_i, orig_idx in enumerate(miss_indices):
            if list_i < len(newly_graded):
                graded_slots[orig_idx] = newly_graded[list_i]
        graded = [g for g in graded_slots if g is not None]

        cache_hits = len(cached_map)
        if cache_hits:
            logger.info(
                f"[Homework] 题库命中 {cache_hits}/{len(questions)} 题，"
                f"LLM 仅批改 {len(miss_questions)} 题"
            )
        logger.info("[Homework] 批改完成")
        _set_grade_progress(
            user.id, request_id, "annotating",
            "逐题判分完成，正在标注知识点与难度…", 78,
        )

        # ── 4. 知识点 + 难度标注 ────────────────────────────────────
        # 这些字段会直接展示在结果页并写入错题本，必须在响应前完成；否则
        # 后台任务即使随后补齐，浏览器已经收到的 JSON 也永远不会更新。
        annotated = graded
        try:
            kp_agent = KnowPointAgent(**model_kw)
            annotated = await kp_agent.process(
                graded, subject=subject_key, known_weak_points=known_weak_points,
                grade_level=user.grade, semester=user.semester,
            )
            logger.info("[Homework] KnowPoint 注解完成")
        except Exception as kp_exc:
            logger.warning(f"[Homework] KnowPoint 失败（降级为无知识点标注）: {kp_exc}")

        # 教材引用只从本次真实命中的 RAG 片段中提取，绝不臆造章节或页码。
        _attach_textbook_references(annotated, textbook_snippets)

        # 把同一大题的题干、图形和所有小问合并成一张原题截图。若视觉模型
        # 未返回可靠区域，则保留整页，完整性优先于裁剪得紧凑。
        try:
            crops = create_question_crops(questions, saved_image_paths, batch_id)
            attach_question_crops(annotated, crops)
        except Exception as crop_exc:
            logger.warning(f"[Homework] 题目截图生成失败（降级为无截图）: {crop_exc}")

        # ── 5. 整份作业标签和薄弱知识点 ──────────────────────────────
        exam_tags: list[str] = []
        weak_points: list[str] = []
        try:
            tag_agent = ExamTagAgent(**model_kw)
            _set_grade_progress(
                user.id, request_id, "summarizing",
                "正在归纳整份作业的考查重点和薄弱知识点…", 87,
            )
            exam_tags, weak_points = await tag_agent.process(
                annotated, subject=subject_key,
                grade_level=user.grade, semester=user.semester,
            )
            logger.info(f"[Homework] ExamTag 完成: tags={exam_tags} weak={weak_points}")
        except Exception as tag_exc:
            logger.warning(f"[Homework] ExamTag 失败（降级为空标签）: {tag_exc}")

        # ── 6. 汇总最终结果 ──────────────────────────────────────────
        result = HomeworkResult(
            subject=subject_key,
            image_count=len(images),
            grade_level=user.grade,
            semester=user.semester,
            textbook_id=curriculum_id,
            used_rag=bool(textbook_snippets),
            textbook_snippets=textbook_snippets[:4],
            exam_tags=exam_tags,
            weak_knowledge_points=weak_points,
            questions=annotated,
        )
        result.compute_stats()
        _set_grade_progress(
            user.id, request_id, "saving",
            "正在保存批改结果、错题本和学生学习档案…", 94,
        )

        logger.info(
            f"[Homework] 完整批改完成 | "
            f"总题数={result.total_questions} "
            f"✓{result.correct_count} "
            f"✗{result.wrong_count} "
            f"△{result.partial_count} "
            f"○{result.blank_count}"
        )

        # ── 7. 自动存档错题（同步，确保含知识点和教材快照）─────────────
        try:
            wb_svc = WrongBookService(user_data_dir(user.id))
            saved = await wb_svc.save_from_result(result, source_image_paths=saved_image_paths)
            if saved:
                logger.info(f"[Homework] 错题本已存档 {saved} 条新错题")
        except Exception as wb_exc:
            logger.warning(f"[Homework] 错题本存档失败（不影响结果）: {wb_exc}")

        record_type_key = record_type.strip().lower()
        if record_type_key not in {"homework", "exam"}:
            record_type_key = "homework"

        # 历史记录先落盘，前端收到“批改完成”后立即刷新即可看到本次记录。
        history_id = await _save_homework_history(result, user.id)

        # 本地结构化画像也必须在响应前更新。这样学生立刻申诉或切换到
        # “我的”页面时，最近成绩已存在，不会发生后台任务竞态。
        try:
            perf = _performance_record_from_result(
                result, record_type_key, exam_name.strip(), history_id or "",
            )
            mem_svc = _get_mem_service(user.id)
            await mem_svc.log_performance(perf)
            current_wrong = await WrongBookService(
                user_data_dir(user.id)
            ).list_entries(mastered=False)
            await mem_svc.reconcile_learning_state(current_wrong)
            logger.info("[Homework] 本地画像已同步")
        except Exception as memory_exc:
            logger.warning(f"[Homework] 本地画像同步失败（不影响批改结果）: {memory_exc}")

        # ── 8. 题库 + EverOS 长期记忆后台持久化 ──────────────────────
        asyncio.create_task(_background_persist_memory(
            newly_graded=newly_graded,   # 仅 LLM 批改的题写入题库
            subject_key=subject_key,
            record_type_key=record_type_key,
            exam_name=exam_name.strip(),
            result=result,
            user_id=user.id,
            curriculum_id=curriculum_id,
        ))

        _set_grade_progress(
            user.id, request_id, "done",
            f"批改完成：共 {result.total_questions} 个小问", 100,
        )

        return result

    except HTTPException:
        _set_grade_progress(
            user.id, request_id, "failed", "批改未完成，请查看页面错误提示", 100,
        )
        raise
    except LLMError as exc:
        logger.error(f"[Homework] AI 模型服务调用失败: {exc}", exc_info=True)
        _set_grade_progress(
            user.id, request_id, "failed", "AI 模型服务暂时不可用", 100,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "AI 模型服务暂时不可用：上游未提供当前配置模型或暂无可用通道。"
                "请稍后重试，或检查 .env 中的 LLM_MODEL 配置"
            ),
        ) from exc
    except Exception as exc:
        logger.error(f"[Homework] 批改过程出错: {exc}", exc_info=True)
        _set_grade_progress(
            user.id, request_id, "failed", "批改服务发生异常", 100,
        )
        raise HTTPException(
            status_code=500,
            detail="批改服务暂时不可用，请稍后重试；如持续失败请查看服务日志",
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# 后台任务：题库 + EverOS（不阻塞批改响应）
# ─────────────────────────────────────────────────────────────────────────────

async def _save_homework_history(result: "HomeworkResult", user_id: str) -> str | None:
    """将完整批改结果持久化到 data/history/homework/{id}.json。"""
    try:
        import json as _json
        from datetime import datetime as _dt
        hist_dir = user_data_dir(user_id) / "history" / "homework"
        hist_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        record_id = f"{ts}_{result.subject}_{uuid.uuid4().hex[:6]}"
        path = hist_dir / f"{record_id}.json"
        atomic_write_text(
            path, _json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
        )
        logger.info(f"[Homework/bg] 批改历史已保存: {record_id}.json")
        return record_id
    except Exception as e:
        logger.warning(f"[Homework/bg] 批改历史保存失败（忽略）: {e}")
        return None


def _performance_record_from_result(
    result: "HomeworkResult", record_type: str, exam_name: str,
    source_history_id: str = "",
) -> PerformanceRecord:
    accuracy = (
        result.correct_count / result.total_questions
        if result.total_questions > 0 else 0.0
    )
    return PerformanceRecord(
        record_type=record_type,
        subject=result.subject,
        exam_name=exam_name,
        source_history_id=source_history_id,
        total_questions=result.total_questions,
        correct_count=result.correct_count,
        wrong_count=result.wrong_count,
        partial_count=result.partial_count,
        blank_count=result.blank_count,
        earned_score=result.earned_score,
        total_score=result.total_score,
        accuracy_rate=round(accuracy, 4),
        weak_knowledge_points=result.weak_knowledge_points,
        exam_tags=result.exam_tags,
    )


def _error_counts(result: "HomeworkResult") -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in result.questions:
        if question.error_type:
            value = (
                question.error_type.value
                if hasattr(question.error_type, "value") else str(question.error_type)
            )
            counts[value] = counts.get(value, 0) + 1
    return counts


async def _background_persist_memory(
    newly_graded: list,   # 仅 LLM 本次批改的（不含题库命中的），写入题库
    subject_key: str,
    record_type_key: str,
    exam_name: str,
    result: "HomeworkResult",
    user_id: str = "",
    curriculum_id: str = "",
) -> None:
    """
    异步后台：题库写入 → 写入 EverOS。
    结果、错题和历史已在响应前保存；增强记忆失败不会影响批改结果。
    """
    # ── 0. 题库写入（只保存本次 LLM 新批改的题）─────────────────────
    try:
        bank = _get_question_bank(curriculum_id)
        saved = await bank.save_batch(newly_graded, subject_key)
        if saved:
            logger.info(f"[Homework/bg] 题库写入 {saved} 道新题")
    except Exception as e:
        logger.warning(f"[Homework/bg] 题库写入失败（忽略）: {e}")

    # ── EverMemOS：长期记忆云端写入 ──────────────────────────────
    try:
        perf_for_evermemos = _performance_record_from_result(
            result, record_type_key, exam_name,
        )
        evermemos = get_evermemos_service(user_id)
        await evermemos.log_performance(perf_for_evermemos)

        error_counts_for_em = _error_counts(result)
        if error_counts_for_em:
            await evermemos.log_error_patterns(subject_key, error_counts_for_em)

        # 每道错题详情写入（帮助 EverMemOS 建立细粒度错题记忆）
        from src.agents.homework.models import GradeResult
        wrong_qs = [
            q for q in result.questions
            if q.grade in (GradeResult.WRONG, GradeResult.PARTIAL)
        ]
        for q in wrong_qs[:10]:   # 最多写 10 道，避免超量
            et = q.error_reason
            if not et and q.error_type and hasattr(q.error_type, "value"):
                et = q.error_type.value
            await evermemos.log_wrong_question(
                subject=subject_key,
                number=q.number,
                question_text=q.question_text,
                student_answer=q.student_answer or "",
                correct_answer=q.correct_answer or "",
                error_type=et,
                brief_comment=q.brief_comment or "",
                knowledge_points=list(q.knowledge_points),
            )

        logger.info(
            f"[Homework/bg] EverMemOS 写入完成 "
            f"（成绩+{len(wrong_qs)}道错题详情）"
        )
    except Exception as e:
        logger.warning(f"[Homework/bg] EverMemOS 写入失败（忽略）: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Regrade (appeal) endpoint
# ─────────────────────────────────────────────────────────────────────────────

class RegradeRequest(BaseModel):
    question_text: str = Field(min_length=1, max_length=12_000)
    student_answer: str = Field(default="", max_length=12_000)
    subject: str = Field(min_length=1, max_length=30)
    question_type: str = Field(default="unknown", max_length=30)
    correct_answer: str = Field(default="", max_length=4_000)
    model_key: Optional[str] = None


@router.post("/regrade-question")
async def regrade_question(body: RegradeRequest, user: CurrentUser = Depends(current_user)):
    """
    **申诉重新评分**：对单道题让 AI 独立重新审阅，返回新的批改结论。

    适用场景：学生认为 AI 判错，希望 AI 重新审题确认是否确实有误。

    返回：
    - `grade`: 新判断结果（correct / wrong / partial / blank / skip）
    - `correct_answer`: 标准答案
    - `brief_comment`: 一句话点评
    - `error_reason`: AI 根据本次实际作答自由判断的错误原因
    - `solution_steps`: 完整解题思路
    - `overturned`: 是否推翻了原判决（新结果为 correct 则为 true）
    """
    if body.subject.strip().lower() not in allowed_subjects(user.grade):
        raise HTTPException(status_code=400, detail=f"未知科目: {body.subject}")

    try:
        subject = body.subject.strip().lower()
        wb_svc = WrongBookService(user_data_dir(user.id))
        entry_id = await wb_svc.find_entry_by_question(
            question_text=body.question_text,
            subject=subject,
            student_answer=body.student_answer,
        )
        entry = await wb_svc.get_entry(entry_id) if entry_id else None

        image_data_url = ""
        if entry and entry.source_image_path:
            image_path = resolve_user_file(entry.source_image_path, user_data_dir(user.id))
            if image_path:
                mime_by_suffix = {
                    ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
                }
                mime = mime_by_suffix.get(image_path.suffix.lower(), "image/jpeg")
                image_data_url = (
                    f"data:{mime};base64,"
                    + base64.b64encode(image_path.read_bytes()).decode()
                )

        grade = entry.grade_level if entry and entry.grade_level else user.grade
        semester = entry.semester if entry and entry.semester else user.semester
        snippets = await _kb_manager.search(
            grade, semester, subject, body.question_text[:800], top_k=4,
        )
        agent = GradeAgent()
        result = await agent.regrade_single(
            question_text=body.question_text,
            student_answer=body.student_answer,
            subject=subject,
            question_type=body.question_type,
            correct_answer=body.correct_answer,
            image_data_url=image_data_url,
            textbook_context="\n\n---\n\n".join(snippets),
        )
        result["overturned"] = result.get("grade") == "correct"
        result["used_original_image"] = bool(image_data_url)
        result["used_rag"] = bool(snippets)
        return result
    except Exception as exc:
        logger.error(f"[Homework] 申诉评分失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="重新评分服务暂时不可用，请稍后重试；如持续失败请查看服务日志",
        ) from exc
