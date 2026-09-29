# Smart Attendance System

[![CI](https://github.com/Lutho123-Pe/Smart-Attendance-System/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Lutho123-Pe/Smart-Attendance-System/actions/workflows/ci.yml)
[![Container](https://github.com/Lutho123-Pe/Smart-Attendance-System/actions/workflows/cd.yml/badge.svg?branch=main)](https://github.com/Lutho123-Pe/Smart-Attendance-System/actions/workflows/cd.yml)

A small, local-first classroom attendance app built with Python, Flask, SQLite, and OpenCV. Face matching is optional and consent-based; the teacher can always mark or correct attendance manually.

## What it does

- Maintains a classroom roster and dated class sessions.
- Lets a teacher enroll up to five face images only after recording explicit opt-in. Images are decoded in memory and are not saved as files.
- Uses OpenCV YuNet face detection and SFace embeddings. Only opted-in templates are candidates for matching; uncertain matches remain unmarked.
- Lets teachers review every session and set `present`, `absent`, or `excused` manually. A manual correction takes precedence over a later automatic match.
- Records consent, recognition, attendance changes, and session events in a local SQLite audit log.
- Supports biometric opt-out at any time from the roster. Opt-out immediately deletes the template; the student remains available for manual attendance.
- Deletes sessions after 180 days, inactive student records after 365 days, and audit events after 365 days by default. These windows are configurable; expired student IDs are also removed from related attendance and audit records.
- Provides a manual-only mode if model files are missing.

This is an educational starter, not an identity-verification or anti-spoofing system. Face matching can be wrong and has no liveness detection. Teachers must review the register, keep a non-biometric option available, and follow their institution's consent, accessibility, security, and data-protection rules. Do not use it for high-stakes decisions or expose it to the public internet without an institution-approved security review.

## Run locally with Docker

1. Install Docker Desktop with Docker Compose.
2. Copy `.env.example` to `.env`. Set a long random `SECRET_KEY` and a unique `TEACHER_PASSWORD`; do not commit `.env`.
3. Start the app:

   ```powershell
   docker compose up --build
   ```

4. Open `http://localhost:5000` and sign in with the teacher password.

Attendance and audit records are stored in the named Docker volume `attendance-data`. Model weights are not included in the image or repository.

## Enable face matching

Python 3.12 and the dependencies in `requirements.txt` are needed to download the models:

```powershell
python -m pip install -r requirements.txt
python scripts/download_models.py
docker compose up --build
```

The downloaded YuNet and SFace files are ignored by Git and mounted read-only into the container. Review the OpenCV Zoo model-card licenses and your institution's rules before use or redistribution. Without both model files, manual attendance still works.

## Run tests

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The tests do not require the face model weights.

## Privacy and operations

- Get and document informed opt-in before any biometric enrollment; never make face matching a condition of attendance.
- Tell students the purpose, what is stored, who can access it, the retention windows, and how to opt out. A teacher records opt-out from the roster; notify students of that route before enrollment.
- Original uploaded images are not retained by the application. Face embeddings and student identifiers are sensitive data; restrict access to the host, protect the `.env`, and secure database backups.
- Attendance/session data expires after `ATTENDANCE_RETENTION_DAYS` (default `180`). Inactive student records and their linked attendance expire after `STUDENT_RETENTION_DAYS` (default `365`). Audit events expire after `AUDIT_RETENTION_DAYS` (default `365`). Cleanup runs when the app starts and can be run with `python -m flask --app 'app:create_app()' cleanup`.
- Face templates are deleted immediately on opt-out. Removing the Docker volume permanently deletes the app database; make and expire backups under the same retention rules.
- The starter uses one shared teacher password. Use a unique secret, enable HTTPS and `SESSION_COOKIE_SECURE=1` behind a trusted TLS proxy, and do not share access. This is not a multi-account system; audit entries identify the actor as `teacher`.

Model similarity thresholds can be tuned with `MATCH_THRESHOLD` and `MATCH_MARGIN`. Validate performance and failure cases with consented, institution-approved images before any classroom trial.

## GitHub CI/CD

- `.github/workflows/ci.yml` runs the test suite on every push and pull request.
- `.github/workflows/cd.yml` reruns tests, then builds and publishes a container to GitHub Container Registry (`ghcr.io`) on pushes to `main` and `v*.*.*` tags. It uses the built-in `GITHUB_TOKEN`; no personal access token is required.
- To open a pull request, create a feature branch and push it. Review and merge only after its CI check passes. For stronger enforcement, enable branch protection on `main` and require the `test` check before merging.
- The first package may be private by default. Adjust the package visibility in GitHub if you intend to showcase or distribute the image publicly.

## Configuration

See `.env.example`. Model paths and matching thresholds can also be overridden with `DETECTOR_MODEL`, `RECOGNIZER_MODEL`, `MATCH_THRESHOLD`, and `MATCH_MARGIN`.

## License

MIT. See [LICENSE](LICENSE).