# GravTraffic (C-01) -- Multi-stage Docker build
#
# Build:
#   docker build -t gravtraffic .
#   docker build --build-arg INSTALL_GPU=1 -t gravtraffic:gpu .
#
# Run:
#   docker run -p 8000:8000 gravtraffic

# ---- Builder stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir setuptools>=68.0 wheel

# Copy project files
COPY pyproject.toml .
COPY gravtraffic/ gravtraffic/

# Install the package with API dependencies
ARG INSTALL_GPU=0
RUN pip install --no-cache-dir --prefix=/install ".[api]" && \
    if [ "$INSTALL_GPU" = "1" ]; then \
        pip install --no-cache-dir --prefix=/install ".[gpu]"; \
    fi

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY gravtraffic/ gravtraffic/
COPY pyproject.toml .

# Non-root user for security
RUN useradd --create-home --shell /bin/bash gravuser
USER gravuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production server: single worker (GravSimulation is not fork-safe)
CMD ["python", "-m", "uvicorn", "gravtraffic.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
