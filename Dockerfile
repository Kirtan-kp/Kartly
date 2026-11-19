# syntax=docker/dockerfile:1
# One image runs the migration, the API and the workers; compose picks the command.

# ---- build: install the package and its dependencies into a virtualenv --------------------
FROM python:3.12-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install .

# ---- runtime: no compiler, no pip cache, no source tree, not root -----------------------
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH
RUN groupadd --system --gid 10001 kartly \
 && useradd --system --uid 10001 --gid kartly --home-dir /app --shell /usr/sbin/nologin kartly
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
# Read at run time relative to /app (KARTLY_MODELS_CONFIG_PATH and KARTLY_KB_DIR default to these).
COPY config ./config
COPY kb ./kb
USER 10001:10001
EXPOSE 8000
# The API binds 0.0.0.0 only inside the container; compose publishes it on 127.0.0.1.
CMD ["python", "-m", "kartly.api", "--host", "0.0.0.0", "--port", "8000"]
