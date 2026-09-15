"""SQLite-backed local profile login and session management."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.domain.curriculum import (
    allowed_subjects,
    profile_grade_label,
    subject_payload,
    validate_grade,
    validate_semester,
)

COOKIE_NAME = "studybuddy_session"
SESSION_DAYS = 30


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _utcnow()).isoformat()


def normalize_name(name: str) -> tuple[str, str]:
    display = " ".join(unicodedata.normalize("NFKC", name).strip().split())
    if not display:
        raise ValueError("姓名不能为空")
    if len(display) > 40:
        raise ValueError("姓名不能超过 40 个字符")
    return display, display.casefold()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CurrentUser:
    id: str
    name: str
    grade: int
    semester: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["grade_label"] = profile_grade_label(self.grade, self.semester)
        data["subjects"] = [subject_payload(key) for key in allowed_subjects(self.grade)]
        return data


class AuthStore:
    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "studybuddy.db"
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    name_key TEXT NOT NULL UNIQUE,
                    grade INTEGER NOT NULL CHECK (grade BETWEEN 1 AND 9),
                    semester TEXT NOT NULL CHECK (semester IN ('upper', 'lower')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE TABLE IF NOT EXISTS textbook_indexes (
                    textbook_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_sha256 TEXT,
                    embedding_model TEXT,
                    status TEXT NOT NULL DEFAULT 'not_indexed',
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )
        self.db_path.chmod(0o600)

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> CurrentUser:
        return CurrentUser(
            id=row["id"], name=row["name"], grade=int(row["grade"]),
            semester=row["semester"], created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def login(self, name: str, grade: int, semester: str) -> tuple[CurrentUser, str, bool]:
        display, name_key = normalize_name(name)
        grade = validate_grade(int(grade))
        semester = validate_semester(semester)
        now = _iso()
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE name_key = ?", (name_key,)).fetchone()
            is_new = row is None
            if row is None:
                user_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (user_id, display, name_key, grade, semester, now, now, now),
                )
            else:
                user_id = row["id"]
                # The login form explicitly supplies the current grade and semester.
                conn.execute(
                    "UPDATE users SET grade=?, semester=?, updated_at=?, last_login_at=? WHERE id=?",
                    (grade, semester, now, now, user_id),
                )
            conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            token = secrets.token_urlsafe(32)
            expires = _iso(_utcnow() + timedelta(days=SESSION_DAYS))
            conn.execute(
                "INSERT INTO sessions(token_hash,user_id,created_at,expires_at) VALUES (?,?,?,?)",
                (_token_hash(token), user_id, now, expires),
            )
            user_row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._row_to_user(user_row), token, is_new

    def get_user_by_token(self, token: str | None) -> CurrentUser | None:
        if not token:
            return None
        now = _iso()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id
                   WHERE s.token_hash=? AND s.expires_at>?""",
                (_token_hash(token), now),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash=?", (_token_hash(token),))

    def update_school(self, user_id: str, grade: int, semester: str) -> CurrentUser:
        grade = validate_grade(int(grade))
        semester = validate_semester(semester)
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE users SET grade=?, semester=?, updated_at=? WHERE id=?",
                (grade, semester, _iso(), user_id),
            )
            row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if row is None:
            raise LookupError("用户不存在")
        return self._row_to_user(row)

    def set_textbook_status(
        self, textbook_id: str, filename: str, status: str, *, source_sha256: str = "",
        embedding_model: str = "", page_count: int = 0, chunk_count: int = 0, error: str = "",
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO textbook_indexes
                   (textbook_id,filename,source_sha256,embedding_model,status,page_count,chunk_count,error,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(textbook_id) DO UPDATE SET
                     filename=excluded.filename, source_sha256=excluded.source_sha256,
                     embedding_model=excluded.embedding_model, status=excluded.status,
                     page_count=excluded.page_count, chunk_count=excluded.chunk_count,
                     error=excluded.error, updated_at=excluded.updated_at""",
                (textbook_id, filename, source_sha256, embedding_model, status,
                 page_count, chunk_count, error[:1000], _iso()),
            )

    def get_textbook_status(self, textbook_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM textbook_indexes WHERE textbook_id=?", (textbook_id,)
            ).fetchone()
        return dict(row) if row else None


_store: AuthStore | None = None


def get_auth_store() -> AuthStore:
    global _store
    if _store is None:
        root = Path(__file__).resolve().parent.parent.parent
        _store = AuthStore(root / "data")
    return _store


def user_data_dir(user_id: str) -> Path:
    root = Path(__file__).resolve().parent.parent.parent / "data" / "users" / user_id
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    return root
