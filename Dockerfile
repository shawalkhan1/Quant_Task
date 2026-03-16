FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    PYTHONPATH=/app

WORKDIR ${APP_HOME}

# Base OS tools for HTTPS and DNS diagnostics.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Create non-root user and writable runtime directories.
RUN useradd -m -u 10001 appuser \
    && mkdir -p /app/data /app/results /app/logs \
    && chown -R appuser:appuser /app

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python docker/healthcheck.py

ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit", "run", "frontend/app.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501"]
