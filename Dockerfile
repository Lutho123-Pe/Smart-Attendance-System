FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATABASE_PATH=/app/instance/attendance.sqlite3

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py recognition.py ./
COPY templates ./templates
COPY static ./static
RUN useradd --system --uid 10001 --no-create-home appuser \
    && install -d -o 10001 -g 10001 /app/instance /app/models

USER 10001:10001
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:5000/healthz', timeout=2)"
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5000", "--access-logfile", "-", "app:create_app()"]