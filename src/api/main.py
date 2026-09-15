"""
StudyAgent AI FastAPI Application
================================

启动方式：
    python src/api/run_server.py

路由前缀 /api/v1
    /homework   — 作业批改（OCR → 批改 → 知识点 → 试卷标签）
    /wrong-book — 错题本（增删改查 + 统计）
    /explain    — 难题解析（所有当前科目均使用对应教材 RAG + GPT-5.5）
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routers import auth, explain, history, homework, memory, textbook, wrong_book
from src.logging import get_logger
from src.services.auth import COOKIE_NAME, get_auth_store

logger = get_logger("API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理（启动 + 关闭）。"""
    logger.info("StudyAgent AI API starting up")

    # 预热 LLM 客户端，让 .env 中的配置提前加载到环境变量
    try:
        from src.services.llm import get_llm_client

        llm_client = get_llm_client()
        logger.info(f"LLM client ready: model={llm_client.config.model}")
    except Exception as e:
        logger.warning(f"LLM client init failed (will retry on first request): {e}")

    yield

    logger.info("StudyAgent AI API shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="StudyAgent AI API",
    description="一年级至九年级学习助手 — 视觉作业批改 / 教材知识库 / 长期记忆",
    version="1.0.0",
    lifespan=lifespan,
    # 避免 HTTPS 反代时 307 重定向降级为 HTTP
    redirect_slashes=False,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_login(request: Request, call_next):
    """Protect all application APIs while leaving login and health reachable."""
    path = request.url.path.rstrip("/")
    host = request.url.hostname
    # 对外只保留 localhost 这一种浏览器来源，避免同一学生出现两套 Cookie、
    # 登录状态和缓存。服务进程仍绑定 127.0.0.1，以确保不能从局域网访问。
    if host not in {"localhost", "127.0.0.1", "testserver"}:
        response = JSONResponse(status_code=400, content={"detail": "仅允许从本机访问"})
    elif host == "127.0.0.1":
        port = f":{request.url.port}" if request.url.port else ""
        target = request.url.replace(netloc=f"localhost{port}")
        response = RedirectResponse(str(target), status_code=308)
    elif (
        request.method != "OPTIONS"
        and path.startswith("/api/v1")
        and path not in {"/api/v1/auth/login", "/api/v1/health"}
    ):
        user = get_auth_store().get_user_by_token(request.cookies.get(COOKIE_NAME))
        if user is None:
            response = JSONResponse(status_code=401, content={"detail": "请先登录"})
        else:
            request.state.user = user
            response = await call_next(request)
    else:
        response = await call_next(request)

    # 这是单机动态应用。尤其是教材 PDF 的路径会随登录档案变化，任何
    # WebView、Service Worker 或浏览器 PDF 查看器都不应复用旧响应。
    if path.startswith("/api/v1") or path.startswith("/ui"):
        response.headers["Cache-Control"] = "private, no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com "
        "https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net "
        "https://fonts.googleapis.com; font-src 'self' https://cdn.jsdelivr.net "
        "https://fonts.gstatic.com data:; img-src 'self' data: blob:; connect-src 'self'; "
        "object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    return response

# ── 运行时数据目录（用户作业图片不做公开静态挂载）──────────────────────────
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# ── 前端静态文件 ──────────────────────────────────────────────────────────────
web_dir = project_root / "web"
web_dir.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(homework.router,    prefix="/api/v1/homework",    tags=["homework"])
app.include_router(wrong_book.router,  prefix="/api/v1/wrong-book",  tags=["wrong-book"])
app.include_router(explain.router,     prefix="/api/v1/explain",     tags=["explain"])
app.include_router(memory.router,      prefix="/api/v1/profile",     tags=["profile"])
app.include_router(history.router,     prefix="/api/v1/history",     tags=["history"])
app.include_router(textbook.router,                                   tags=["textbook"])
# knowledge router 暂不注册（依赖 src.api.utils，StudyAgent AI 尚未实现）
# app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Welcome to StudyAgent AI API", "version": "1.0.0"}


@app.get("/api/v1/health")
async def health():
    """全局健康检查。"""
    return {"status": "ok"}
