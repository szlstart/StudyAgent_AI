from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.api import main as main_module
from src.api.routers import auth as auth_router
from src.api.routers import memory as memory_router
from src.api.routers import history as history_router
from src.api.routers import homework as homework_router
from src.api.routers import wrong_book as wrong_book_router
from src.domain.curriculum import allowed_subjects, textbook_filename
from src.services import auth as auth_service
from src.services.auth import AuthStore


class _NoopEverMemOS:
    async def log_user_profile(self, _profile):
        return None


def test_curriculum_is_grade_specific():
    assert allowed_subjects(1) == ("chinese", "math")
    assert allowed_subjects(7) == (
        "chinese", "math", "english", "biology", "history", "geography"
    )
    assert allowed_subjects(9) == (
        "chinese", "math", "english", "physics", "history", "chemistry"
    )
    assert textbook_filename(7, "lower", "biology") == "生物_七年级_下册.pdf"


def test_loopback_ip_redirects_to_canonical_localhost():
    client = TestClient(
        main_module.app,
        base_url="http://127.0.0.1:8001",
        follow_redirects=False,
    )
    response = client.get("/ui/?from=test")
    assert response.status_code == 308
    assert response.headers["location"] == "http://localhost:8001/ui/?from=test"
    assert response.headers["x-frame-options"] == "DENY"


def test_non_loopback_host_is_rejected():
    client = TestClient(main_module.app, base_url="http://attacker.invalid")
    response = client.get("/api/v1/health")
    assert response.status_code == 400
    assert response.json()["detail"] == "仅允许从本机访问"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_name_is_unique_and_reuses_the_same_profile(tmp_path):
    store = AuthStore(tmp_path / "db")
    first, _, is_new = store.login("李雷", 7, "upper")
    again, _, is_new_again = store.login("  李雷  ", 8, "lower")
    assert is_new is True
    assert is_new_again is False
    assert again.id == first.id
    assert again.name == "李雷"
    assert (again.grade, again.semester) == (8, "lower")


def test_api_requires_login_and_isolates_profiles(tmp_path, monkeypatch):
    store = AuthStore(tmp_path / "db")
    users_root = tmp_path / "users"

    def local_user_dir(user_id: str):
        path = users_root / user_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(main_module, "get_auth_store", lambda: store)
    monkeypatch.setattr(auth_router, "get_auth_store", lambda: store)
    monkeypatch.setattr(auth_router, "user_data_dir", local_user_dir)
    monkeypatch.setattr(memory_router, "user_data_dir", local_user_dir)
    monkeypatch.setattr(history_router, "user_data_dir", local_user_dir)
    monkeypatch.setattr(homework_router, "user_data_dir", local_user_dir)
    monkeypatch.setattr(wrong_book_router, "user_data_dir", local_user_dir)
    monkeypatch.setattr(auth_service, "user_data_dir", local_user_dir)
    monkeypatch.setattr(auth_router, "get_evermemos_service", lambda _uid: _NoopEverMemOS())
    memory_router._mem_services.clear()
    wrong_book_router._services.clear()

    anonymous = TestClient(main_module.app)
    assert anonymous.get("/api/v1/profile").status_code == 401
    assert "no-store" in anonymous.get("/ui/").headers["cache-control"]

    alice = TestClient(main_module.app)
    bob = TestClient(main_module.app)
    response = alice.post(
        "/api/v1/auth/login", json={"name": "小雨", "grade": 7, "semester": "lower"}
    )
    assert response.status_code == 200
    alice_id = response.json()["user"]["id"]
    assert [s["key"] for s in response.json()["user"]["subjects"]] == list(allowed_subjects(7))
    grade7_status = alice.get("/api/v1/textbook/status").json()
    assert grade7_status["grade_label"] == "七年级下册"
    grade7_chinese = next(row for row in grade7_status["textbooks"] if row["subject"] == "chinese")
    grade7_file = alice.get(
        f"/api/v1/textbook/file/chinese/{grade7_chinese['pdf_filename']}",
        params={"textbook": grade7_chinese["textbook_id"]},
    )
    assert grade7_file.status_code == 200
    assert grade7_file.headers["x-studybuddy-textbook"] == "grade_07_lower_chinese"
    assert "no-store" in grade7_file.headers["cache-control"]

    assert bob.post(
        "/api/v1/auth/login", json={"name": "小林", "grade": 3, "semester": "upper"}
    ).status_code == 200

    # 错题和历史文件也必须按用户目录隔离，不能只隔离个人偏好。
    alice_dir = local_user_dir(alice_id)
    (alice_dir / "wrong_book").mkdir(parents=True, exist_ok=True)
    (alice_dir / "wrong_book" / "entries.json").write_text(json.dumps([{
        "entry_id": "math_20260903_p1_1", "subject": "math",
        "question_text": "1+1=?", "student_answer": "3", "correct_answer": "2",
        "question_type": "fill_blank", "grade": "wrong",
        "source_image_path": str(alice_dir / "homework_images" / "private.jpg"),
    }]), encoding="utf-8")
    (alice_dir / "homework_images").mkdir(parents=True, exist_ok=True)
    (alice_dir / "homework_images" / "private.jpg").write_bytes(b"private-image")
    (alice_dir / "history" / "homework").mkdir(parents=True, exist_ok=True)
    (alice_dir / "history" / "homework" / "20260903_120000_math.json").write_text(
        json.dumps({"subject": "math", "total_questions": 1, "correct_count": 0}),
        encoding="utf-8",
    )
    alice_wrong = alice.get("/api/v1/wrong-book").json()
    assert len(alice_wrong) == 1
    assert "source_image_path" not in alice_wrong[0]
    assert alice_wrong[0]["question_image_urls"] == [
        "/api/v1/homework/images/private.jpg"
    ]
    assert alice.get("/api/v1/homework/images/private.jpg").content == b"private-image"
    assert bob.get("/api/v1/homework/images/private.jpg").status_code == 404
    assert alice.get("/api/v1/homework/images/../studybuddy.db").status_code == 404
    assert bob.get("/api/v1/wrong-book").json() == []
    assert len(alice.get("/api/v1/history/homework").json()) == 1
    assert bob.get("/api/v1/history/homework").json() == []
    assert alice.patch("/api/v1/profile/preferences", json={"mbti": "INTP"}).status_code == 200
    assert bob.patch("/api/v1/profile/preferences", json={"mbti": "ISFJ"}).status_code == 200
    assert alice.get("/api/v1/profile").json()["preferences"]["mbti"] == "INTP"
    assert bob.get("/api/v1/profile").json()["preferences"]["mbti"] == "ISFJ"
    assert alice.patch("/api/v1/profile/preferences", json={"mbti": "XXXX"}).status_code == 400

    changed = alice.patch(
        "/api/v1/auth/profile", json={"name": "不能改名", "grade": 8, "semester": "upper"}
    )
    assert changed.status_code == 200
    assert changed.json()["user"]["name"] == "小雨"
    assert changed.json()["user"]["grade_label"] == "八年级上册"
    # 上一个年级的 PDF 链接不能在新档案下静默返回另一册教材。
    stale_file = alice.get(
        "/api/v1/textbook/file/chinese",
        params={"textbook": grade7_chinese["textbook_id"]},
    )
    assert stale_file.status_code == 409
    grade8_status = alice.get("/api/v1/textbook/status").json()
    grade8_chinese = next(row for row in grade8_status["textbooks"] if row["subject"] == "chinese")
    grade8_file = alice.get(
        f"/api/v1/textbook/file/chinese/{grade8_chinese['pdf_filename']}",
        params={"textbook": grade8_chinese["textbook_id"]},
    )
    assert grade8_file.status_code == 200
    assert grade8_file.headers["x-studybuddy-textbook"] == "grade_08_upper_chinese"
    wrong_filename = alice.get(
        f"/api/v1/textbook/file/chinese/{grade7_chinese['pdf_filename']}",
        params={"textbook": grade8_chinese["textbook_id"]},
    )
    assert wrong_filename.status_code == 409
