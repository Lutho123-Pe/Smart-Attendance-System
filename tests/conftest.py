import re

import pytest

from app import create_app


@pytest.fixture
def app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "TEACHER_PASSWORD": "test-password",
            "DATABASE": str(tmp_path / "attendance.sqlite3"),
            "DETECTOR_MODEL": str(tmp_path / "missing-detector.onnx"),
            "RECOGNIZER_MODEL": str(tmp_path / "missing-recognizer.onnx"),
            "ATTENDANCE_RETENTION_DAYS": 180,
            "AUDIT_RETENTION_DAYS": 365,
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


def login(client):
    page = client.get("/login")
    token = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True)).group(1)
    return client.post("/login", data={"csrf_token": token, "password": "test-password"})


@pytest.fixture
def teacher(client):
    login(client)
    return client


@pytest.fixture
def login_helper():
    return login