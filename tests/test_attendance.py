import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np

from app import cleanup_expired_data


def form_token(client, path="/"):
    response = client.get(path)
    assert response.status_code == 200
    import re

    return re.search(r'name="csrf_token" value="([^"]+)"', response.get_data(as_text=True)).group(1)


def test_manual_attendance_and_correction_are_audited(app, teacher):
    token = form_token(teacher)
    teacher.post("/students", data={"csrf_token": token, "student_code": "S-01", "name": "Ari Example"})
    token = form_token(teacher)
    response = teacher.post("/sessions", data={"csrf_token": token, "course": "Biology 204"})
    assert response.status_code == 302
    session_id = int(response.headers["Location"].rsplit("/", 1)[1])

    with sqlite3.connect(app.config["DATABASE"]) as db:
        student_id = db.execute("SELECT id FROM students WHERE student_code = 'S-01'").fetchone()[0]
    token = form_token(teacher, f"/sessions/{session_id}")
    teacher.post(
        f"/sessions/{session_id}/attendance",
        data={"csrf_token": token, "student_id": student_id, "status": "absent", "note": "Not yet checked"},
    )
    token = form_token(teacher, f"/sessions/{session_id}")
    teacher.post(
        f"/sessions/{session_id}/attendance",
        data={"csrf_token": token, "student_id": student_id, "status": "present", "note": "Arrived early"},
    )

    with sqlite3.connect(app.config["DATABASE"]) as db:
        attendance = db.execute("SELECT status, source, note FROM attendance").fetchone()
        event_count = db.execute(
            "SELECT COUNT(*) FROM audit_log WHERE action = 'attendance_corrected'"
        ).fetchone()[0]
    assert attendance == ("present", "manual", "Arrived early")
    assert event_count == 2


def test_face_images_without_explicit_consent_are_rejected(app, teacher):
    token = form_token(teacher, "/students")
    response = teacher.post(
        "/students",
        data={
            "csrf_token": token,
            "student_code": "S-02",
            "name": "Sam Example",
            "photos": (bytes([1, 2, 3]), "portrait.jpg"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    with sqlite3.connect(app.config["DATABASE"]) as db:
        count = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    assert count == 0


def test_opt_out_deletes_template_and_keeps_manual_roster(app, teacher):
    with sqlite3.connect(app.config["DATABASE"]) as db:
        db.execute(
            "INSERT INTO students(student_code, name, consented, embedding, created_at) VALUES (?, ?, 1, ?, ?)",
            ("S-03", "Lee Example", bytes([0, 0, 128, 63]), "2026-01-01T00:00:00+00:00"),
        )
        student_id = db.execute("SELECT id FROM students WHERE student_code = 'S-03'").fetchone()[0]
    token = form_token(teacher, "/students")
    response = teacher.post(f"/students/{student_id}/opt-out", data={"csrf_token": token})
    assert response.status_code == 302
    with sqlite3.connect(app.config["DATABASE"]) as db:
        row = db.execute("SELECT consented, opted_out, embedding FROM students WHERE id = ?", (student_id,)).fetchone()
        action = db.execute("SELECT action FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert row == (0, 1, None)
    assert action == "biometric_opt_out"


def test_student_can_later_opt_in_with_fresh_consent(app, teacher):
    with sqlite3.connect(app.config["DATABASE"]) as db:
        db.execute(
            "INSERT INTO students(student_code, name, created_at) VALUES (?, ?, ?)",
            ("S-04", "Pat Example", "2026-01-01T00:00:00+00:00"),
        )
        student_id = db.execute("SELECT id FROM students WHERE student_code = 'S-04'").fetchone()[0]

    class FakeEngine:
        @staticmethod
        def embedding(_image):
            return np.array([1.0, 0.0], dtype=np.float32)

        @staticmethod
        def average(vectors):
            return vectors[0]

    app.extensions["face_engine"] = FakeEngine()
    token = form_token(teacher, "/students")
    response = teacher.post(
        f"/students/{student_id}/enroll",
        data={
            "csrf_token": token,
            "consent": "yes",
            "photos": (b"temporary image bytes", "portrait.jpg"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    with sqlite3.connect(app.config["DATABASE"]) as db:
        row = db.execute(
            "SELECT consented, opted_out, embedding, consented_at FROM students WHERE id = ?", (student_id,)
        ).fetchone()
        action = db.execute("SELECT action FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert row[0:2] == (1, 0)
    assert row[2] == np.array([1.0, 0.0], dtype=np.float32).tobytes()
    assert row[3]
    assert action == "biometric_opt_in"


def test_inactive_student_records_expire_with_linked_identifiers(app):
    expired_at = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    with sqlite3.connect(app.config["DATABASE"]) as db:
        db.execute(
            """INSERT INTO students(student_code, name, created_at, last_activity_at)
               VALUES (?, ?, ?, ?)""",
            ("OLD-01", "Expired Example", expired_at, expired_at),
        )
        student_id = db.execute("SELECT id FROM students WHERE student_code = 'OLD-01'").fetchone()[0]
        db.execute(
            "INSERT INTO class_sessions(course, held_at, created_at) VALUES (?, ?, ?)",
            ("Current class", datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()),
        )
        session_id = db.execute("SELECT id FROM class_sessions").fetchone()[0]
        db.execute(
            "INSERT INTO attendance(class_session_id, student_id, status, source, updated_at) VALUES (?, ?, 'present', 'manual', ?)",
            (session_id, student_id, datetime.now(timezone.utc).isoformat()),
        )
        db.execute(
            "INSERT INTO audit_log(actor, action, target, created_at) VALUES ('teacher', 'student_added', 'OLD-01', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
    with app.app_context():
        cleanup_expired_data()
    with sqlite3.connect(app.config["DATABASE"]) as db:
        assert db.execute("SELECT COUNT(*) FROM students WHERE student_code = 'OLD-01'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ?", (student_id,)).fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM audit_log WHERE target = 'OLD-01'").fetchone()[0] == 0


def test_manual_attendance_works_without_models(teacher):
    page = teacher.get("/")
    assert b"Manual mode" in page.data
    assert page.status_code == 200


def test_manual_absence_is_not_overwritten_by_later_face_match(app, teacher):
    with sqlite3.connect(app.config["DATABASE"]) as db:
        db.execute(
            """INSERT INTO students(student_code, name, consented, embedding, created_at)
               VALUES (?, ?, 1, ?, ?)""",
            ("S-05", "Casey Example", np.array([1.0, 0.0], dtype=np.float32).tobytes(), "2026-01-01T00:00:00+00:00"),
        )
        student_id = db.execute("SELECT id FROM students WHERE student_code = 'S-05'").fetchone()[0]
        db.execute(
            "INSERT INTO class_sessions(course, held_at, created_at) VALUES (?, ?, ?)",
            ("Chemistry 101", "2026-09-29T09:00", "2026-09-29T09:00:00+00:00"),
        )
        session_id = db.execute("SELECT id FROM class_sessions").fetchone()[0]

    class FakeEngine:
        @staticmethod
        def recognize(_image, _templates):
            return [{"student_id": student_id, "score": 0.99}]

    app.extensions["face_engine"] = FakeEngine()
    token = form_token(teacher, f"/sessions/{session_id}")
    teacher.post(
        f"/sessions/{session_id}/attendance",
        data={"csrf_token": token, "student_id": student_id, "status": "absent", "note": "Teacher confirmed absence"},
    )
    token = form_token(teacher, f"/sessions/{session_id}")
    teacher.post(
        f"/sessions/{session_id}/recognize",
        data={"csrf_token": token, "snapshot": (b"temporary classroom image", "class.jpg")},
        content_type="multipart/form-data",
    )

    with sqlite3.connect(app.config["DATABASE"]) as db:
        status, source = db.execute(
            "SELECT status, source FROM attendance WHERE class_session_id = ? AND student_id = ?",
            (session_id, student_id),
        ).fetchone()
    assert (status, source) == ("absent", "manual")


def test_all_mutations_require_csrf(client):
    response = client.post("/login", data={"password": "test-password"})
    assert response.status_code == 400


def test_teacher_pages_require_login(client):
    response = client.get("/")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]