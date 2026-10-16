
FROM python:3.13-slim

WORKDIR /app

# Install uv from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure Python and the virtual environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Copy dependency definitions first for build caching
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies using the existing lockfile
RUN uv sync --frozen --no-install-project

# Copy application source code
COPY . .

# Document the FastAPI port
EXPOSE 8000

# Default command: start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]