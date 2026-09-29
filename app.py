import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import numpy as np
from flask import (
    Flask,
    abort,
    current_app,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from recognition import FaceEngine, ModelUnavailable


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    consented INTEGER NOT NULL DEFAULT 0,
    consented_at TEXT,
    opted_out INTEGER NOT NULL DEFAULT 0,
    opted_out_at TEXT,
    embedding BLOB,
    created_at TEXT NOT NULL,
    last_activity_at TEXT
);
CREATE TABLE IF NOT EXISTS class_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course TEXT NOT NULL,
    held_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_session_id INTEGER NOT NULL REFERENCES class_sessions(id) ON DELETE CASCADE,
    student_id INTEGER NOT NULL REFERENCES students(id),
    status TEXT NOT NULL CHECK(status IN ('present', 'absent', 'excused')),
    source TEXT NOT NULL CHECK(source IN ('face', 'manual')),
    note TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    UNIQUE(class_session_id, student_id)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_db():
    if "db" not in g:
        database_path = Path(current_app.config["DATABASE"])
        database_path.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(database_path)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def audit(actor, action, target, details=None):
    get_db().execute(
        "INSERT INTO audit_log(actor, action, target, details, created_at) VALUES (?, ?, ?, ?, ?)",
        (actor, action, target, json.dumps(details or {}, separators=(",", ":")), utc_now()),
    )


def cleanup_expired_data():
    db = get_db()
    now = datetime.now(timezone.utc)
    attendance_cutoff = (now - timedelta(days=current_app.config["ATTENDANCE_RETENTION_DAYS"])).isoformat()
    audit_cutoff = (now - timedelta(days=current_app.config["AUDIT_RETENTION_DAYS"])).isoformat()
        student_cutoff = (now - timedelta(days=current_app.config["STUDENT_RETENTION_DAYS"])).isoformat()
    db.execute("DELETE FROM class_sessions WHERE created_at < ?", (attendance_cutoff,))
        db.execute(
                """DELETE FROM audit_log WHERE action IN (
                         'student_added', 'biometric_opt_in', 'biometric_opt_out',
                         'attendance_corrected', 'attendance_marked'
                     ) AND target IN (
                         SELECT student_code FROM students
                         WHERE COALESCE(last_activity_at, created_at) < ?
                     )""",
                (student_cutoff,),
        )
        db.execute(
                """DELETE FROM attendance WHERE student_id IN (
                         SELECT id FROM students WHERE COALESCE(last_activity_at, created_at) < ?
                     )""",
                (student_cutoff,),
        )
        db.execute("DELETE FROM students WHERE COALESCE(last_activity_at, created_at) < ?", (student_cutoff,))
    db.execute("DELETE FROM audit_log WHERE created_at < ?", (audit_cutoff,))
    db.commit()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("teacher"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def get_face_engine():
    if "face_engine" not in current_app.extensions:
        try:
            current_app.extensions["face_engine"] = FaceEngine(
                current_app.config["DETECTOR_MODEL"],
                current_app.config["RECOGNIZER_MODEL"],
                current_app.config["MATCH_THRESHOLD"],
                current_app.config["MATCH_MARGIN"],
            )
        except ModelUnavailable:
            current_app.extensions["face_engine"] = None
    engine = current_app.extensions["face_engine"]
    if engine is None:
        raise ModelUnavailable("Face models are not installed; manual attendance is available.")
    return engine


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", ""),
        TEACHER_PASSWORD=os.environ.get("TEACHER_PASSWORD", ""),
        DATABASE=os.environ.get("DATABASE_PATH", "instance/attendance.sqlite3"),
        DETECTOR_MODEL=os.environ.get("DETECTOR_MODEL", "models/face_detection_yunet_2023mar.onnx"),
        RECOGNIZER_MODEL=os.environ.get("RECOGNIZER_MODEL", "models/face_recognition_sface_2021dec.onnx"),
        MATCH_THRESHOLD=float(os.environ.get("MATCH_THRESHOLD", "0.40")),
        MATCH_MARGIN=float(os.environ.get("MATCH_MARGIN", "0.05")),
        ATTENDANCE_RETENTION_DAYS=int(os.environ.get("ATTENDANCE_RETENTION_DAYS", "180")),
        AUDIT_RETENTION_DAYS=int(os.environ.get("AUDIT_RETENTION_DAYS", "365")),
        STUDENT_RETENTION_DAYS=int(os.environ.get("STUDENT_RETENTION_DAYS", "365")),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "0") == "1",
    )
    if test_config:
        app.config.update(test_config)
    if not app.config["SECRET_KEY"] or not app.config["TEACHER_PASSWORD"]:
        raise RuntimeError("Set SECRET_KEY and TEACHER_PASSWORD before starting the app.")

    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(app.config["DATABASE"]) as db:
        db.executescript(SCHEMA)
        student_columns = {row[1] for row in db.execute("PRAGMA table_info(students)")}
        if "consented_at" not in student_columns:
            db.execute("ALTER TABLE students ADD COLUMN consented_at TEXT")
        if "opted_out_at" not in student_columns:
            db.execute("ALTER TABLE students ADD COLUMN opted_out_at TEXT")
        if "last_activity_at" not in student_columns:
            db.execute("ALTER TABLE students ADD COLUMN last_activity_at TEXT")
    with app.app_context():
        cleanup_expired_data()

    @app.teardown_appcontext
    def close_db(_error=None):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.before_request
    def csrf_protect():
        if request.method == "POST":
            submitted = request.form.get("csrf_token", "")
            expected = session.get("csrf_token", "")
            if not expected or not hmac.compare_digest(submitted, expected):
                abort(400, "Invalid form token. Reload the page and try again.")

    @app.context_processor
    def inject_form_token():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        return {"csrf_token": session["csrf_token"]}

    @app.get("/login")
    def login():
        return render_template("login.html")

    @app.post("/login")
    def login_post():
        if not hmac.compare_digest(request.form.get("password", ""), app.config["TEACHER_PASSWORD"]):
            flash("Password not recognized.", "error")
            return render_template("login.html"), 401
        session.clear()
        session["teacher"] = "teacher"
        session["csrf_token"] = secrets.token_urlsafe(32)
        return redirect(url_for("index"))

    @app.post("/logout")
    @login_required
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    @login_required
    def index():
        db = get_db()
        sessions = db.execute(
            """SELECT c.*, COUNT(CASE WHEN a.status = 'present' THEN 1 END) AS present_count
               FROM class_sessions c LEFT JOIN attendance a ON a.class_session_id = c.id
               GROUP BY c.id ORDER BY c.held_at DESC, c.id DESC LIMIT 12"""
        ).fetchall()
        student_count = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        opted_in_count = db.execute(
            "SELECT COUNT(*) FROM students WHERE consented = 1 AND opted_out = 0 AND embedding IS NOT NULL"
        ).fetchone()[0]
        model_ready = Path(app.config["DETECTOR_MODEL"]).is_file() and Path(
            app.config["RECOGNIZER_MODEL"]
        ).is_file()
        return render_template(
            "index.html",
            sessions=sessions,
            student_count=student_count,
            opted_in_count=opted_in_count,
            model_ready=model_ready,
        )

    @app.get("/students")
    @login_required
    def students():
        roster = get_db().execute("SELECT * FROM students ORDER BY name COLLATE NOCASE").fetchall()
        return render_template("students.html", students=roster)

    @app.post("/students")
    @login_required
    def add_student():
        code = request.form.get("student_code", "").strip()
        name = request.form.get("name", "").strip()
        photos = [photo for photo in request.files.getlist("photos") if photo.filename]
        consented = request.form.get("consent") == "yes"
        if not code or not name or len(code) > 40 or len(name) > 100:
            flash("Enter a student ID (up to 40 characters) and name (up to 100 characters).", "error")
            return redirect(url_for("students"))
        if photos and not consented:
            flash("Explicit biometric consent is required before enrolling face images.", "error")
            return redirect(url_for("students"))
        if len(photos) > 5:
            flash("Use at most five consented images per student.", "error")
            return redirect(url_for("students"))
        embedding = None
        if photos:
            try:
                engine = get_face_engine()
                vectors = [engine.embedding(photo.read()) for photo in photos]
                embedding = engine.average(vectors).tobytes()
            except (ModelUnavailable, ValueError) as error:
                flash(str(error), "error")
                return redirect(url_for("students"))
        db = get_db()
        try:
                created_at = utc_now()
            cursor = db.execute(
                     """INSERT INTO students(student_code, name, consented, consented_at, embedding,
                         created_at, last_activity_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                     (code, name, int(consented and embedding is not None), created_at if embedding else None,
                      embedding, created_at, created_at),
            )
            audit(
                "teacher",
                "student_added",
                code,
                {"biometric_enrolled": bool(embedding), "consent_recorded": bool(embedding and consented)},
            )
            db.commit()
        except sqlite3.IntegrityError:
            db.rollback()
            flash("That student ID is already on the roster.", "error")
            return redirect(url_for("students"))
        flash("Student added to the manual roster." + (" Face template enrolled." if embedding else ""), "success")
        return redirect(url_for("students"))

    @app.post("/students/<int:student_id>/enroll")
    @login_required
    def enroll_student(student_id):
        db = get_db()
        student = db.execute("SELECT student_code FROM students WHERE id = ?", (student_id,)).fetchone()
        if student is None:
            abort(404)
        photos = [photo for photo in request.files.getlist("photos") if photo.filename]
        if request.form.get("consent") != "yes":
            flash("Fresh, explicit biometric consent is required to enroll face images.", "error")
            return redirect(url_for("students"))
        if not photos or len(photos) > 5:
            flash("Choose between one and five consented images.", "error")
            return redirect(url_for("students"))
        try:
            engine = get_face_engine()
            embedding = engine.average([engine.embedding(photo.read()) for photo in photos]).tobytes()
        except (ModelUnavailable, ValueError) as error:
            flash(str(error), "error")
            return redirect(url_for("students"))
        db.execute(
            """UPDATE students SET consented = 1, consented_at = ?, opted_out = 0,
                    opted_out_at = NULL, embedding = ?, last_activity_at = ? WHERE id = ?""",
                (utc_now(), embedding, utc_now(), student_id),
        )
        audit("teacher", "biometric_opt_in", student["student_code"], {
            "consent_recorded": True,
            "consent_version": "face-enrollment-v1",
        })
        db.commit()
        flash("Consent recorded and face template enrolled. Uploaded images were discarded.", "success")
        return redirect(url_for("students"))

    @app.post("/students/<int:student_id>/opt-out")
    @login_required
    def opt_out(student_id):
        db = get_db()
        student = db.execute("SELECT student_code FROM students WHERE id = ?", (student_id,)).fetchone()
        if student is None:
            abort(404)
        db.execute(
            """UPDATE students SET consented = 0, consented_at = NULL, opted_out = 1,
               opted_out_at = ?, embedding = NULL, last_activity_at = ? WHERE id = ?""",
            (utc_now(), utc_now(), student_id),
        )
        audit("teacher", "biometric_opt_out", student["student_code"], {"template_deleted": True})
        db.commit()
        flash("Biometric template deleted. The student remains on the manual roster.", "success")
        return redirect(url_for("students"))

    @app.post("/sessions")
    @login_required
    def create_session():
        course = request.form.get("course", "").strip()
        held_at = request.form.get("held_at", "").strip() or utc_now()
        if not course or len(course) > 100:
            flash("Enter a class name (up to 100 characters).", "error")
            return redirect(url_for("index"))
        db = get_db()
        cursor = db.execute(
            "INSERT INTO class_sessions(course, held_at, created_at) VALUES (?, ?, ?)",
            (course, held_at, utc_now()),
        )
        audit("teacher", "session_created", str(cursor.lastrowid), {"course": course})
        db.commit()
        return redirect(url_for("session_detail", session_id=cursor.lastrowid))

    @app.get("/sessions/<int:session_id>")
    @login_required
    def session_detail(session_id):
        db = get_db()
        class_session = db.execute("SELECT * FROM class_sessions WHERE id = ?", (session_id,)).fetchone()
        if class_session is None:
            abort(404)
        roster = db.execute(
            """SELECT s.id, s.student_code, s.name, s.opted_out, a.status, a.source, a.note
               FROM students s LEFT JOIN attendance a
               ON a.student_id = s.id AND a.class_session_id = ?
               ORDER BY s.name COLLATE NOCASE""",
            (session_id,),
        ).fetchall()
        return render_template("session.html", class_session=class_session, roster=roster)

    @app.post("/sessions/<int:session_id>/attendance")
    @login_required
    def set_attendance(session_id):
        db = get_db()
        if db.execute("SELECT 1 FROM class_sessions WHERE id = ?", (session_id,)).fetchone() is None:
            abort(404)
        student_id = request.form.get("student_id", type=int)
        status = request.form.get("status", "")
        note = request.form.get("note", "").strip()[:240]
        student = db.execute("SELECT student_code FROM students WHERE id = ?", (student_id,)).fetchone()
        if student is None or status not in {"present", "absent", "excused"}:
            abort(400, "Choose a roster student and a valid attendance status.")
        db.execute(
            """INSERT INTO attendance(class_session_id, student_id, status, source, note, updated_at)
               VALUES (?, ?, ?, 'manual', ?, ?)
               ON CONFLICT(class_session_id, student_id) DO UPDATE SET
               status = excluded.status, source = 'manual', note = excluded.note, updated_at = excluded.updated_at""",
            (session_id, student_id, status, note, utc_now()),
        )
        db.execute("UPDATE students SET last_activity_at = ? WHERE id = ?", (utc_now(), student_id))
        audit("teacher", "attendance_corrected", student["student_code"], {
            "session_id": session_id,
            "status": status,
            "note": note,
        })
        db.commit()
        flash("Manual attendance saved.", "success")
        return redirect(url_for("session_detail", session_id=session_id))

    @app.post("/sessions/<int:session_id>/recognize")
    @login_required
    def recognize_snapshot(session_id):
        db = get_db()
        if db.execute("SELECT 1 FROM class_sessions WHERE id = ?", (session_id,)).fetchone() is None:
            abort(404)
        photo = request.files.get("snapshot")
        if photo is None or not photo.filename:
            flash("Choose a classroom snapshot to process.", "error")
            return redirect(url_for("session_detail", session_id=session_id))
        students_with_embeddings = db.execute(
            "SELECT id, student_code, embedding FROM students WHERE consented = 1 AND opted_out = 0 AND embedding IS NOT NULL"
        ).fetchall()
        if not students_with_embeddings:
            flash("No opted-in face templates are available. You can still mark attendance manually.", "error")
            return redirect(url_for("session_detail", session_id=session_id))
        try:
            engine = get_face_engine()
            faces = engine.recognize(
                photo.read(),
                [(row["id"], np.frombuffer(row["embedding"], dtype=np.float32)) for row in students_with_embeddings],
            )
        except (ModelUnavailable, ValueError) as error:
            flash(str(error), "error")
            return redirect(url_for("session_detail", session_id=session_id))

        student_codes = {row["id"]: row["student_code"] for row in students_with_embeddings}
        matched_ids = set()
        new_matches = 0
        for face in faces:
            student_id = face["student_id"]
            if student_id is None or student_id in matched_ids:
                continue
            matched_ids.add(student_id)
            db.execute(
                """INSERT INTO attendance(class_session_id, student_id, status, source, note, updated_at)
                   VALUES (?, ?, 'present', 'face', '', ?)
                   ON CONFLICT(class_session_id, student_id) DO NOTHING""",
                (session_id, student_id, utc_now()),
            )
            if db.execute("SELECT changes()").fetchone()[0]:
                new_matches += 1
                audit("recognition", "attendance_marked", student_codes[student_id], {
                    "session_id": session_id,
                    "similarity": round(face["score"], 4),
                })
        if matched_ids:
            db.executemany(
                "UPDATE students SET last_activity_at = ? WHERE id = ?",
                [(utc_now(), student_id) for student_id in matched_ids],
            )
        audit("teacher", "snapshot_processed", str(session_id), {
            "face_count": len(faces),
            "new_matches": new_matches,
        })
        db.commit()
        flash(f"Snapshot processed: {len(faces)} face(s) detected, {new_matches} new match(es). Review the roster below.", "success")
        return redirect(url_for("session_detail", session_id=session_id))

    @app.get("/audit")
    @login_required
    def audit_page():
        entries = get_db().execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 150").fetchall()
        return render_template("audit.html", entries=entries)

    @app.get("/healthz")
    def health_check():
        return "ok", 200, {"Content-Type": "text/plain; charset=utf-8"}

    @app.cli.command("cleanup")
    def cleanup_command():
        cleanup_expired_data()
        print("Expired attendance sessions and audit events removed.")

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))