# Build from the repository root: docker build -f apps/api/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY apps/api ./apps/api

# Historical V1/V2 artifacts in the project Drive use sklearn 1.6.1.
# Match that release during unpickling rather than relying on latest sklearn.
RUN python -m pip install --no-cache-dir -e ".[api]" \
    "scikit-learn==1.6.1" "joblib>=1.3,<2"

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
