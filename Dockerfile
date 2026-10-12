ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm
FROM ${PYTHON_IMAGE} AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY openagent ./openagent
COPY web ./web
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM ${PYTHON_IMAGE} AS runtime

ARG OPENAGENT_VERSION=0.13.0
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="OpenAgent" \
      org.opencontainers.image.description="Self-hosted OpenAgent runtime" \
      org.opencontainers.image.source="https://github.com/awais-akhtar/OpenAgent" \
      org.opencontainers.image.version="${OPENAGENT_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENAGENT_HOME=/data \
    OPENAGENT_DEPLOYMENT_MODE=team \
    OPENAGENT_AUTH_ENFORCED=1

RUN addgroup --system --gid 10001 openagent \
    && adduser --system --uid 10001 --ingroup openagent --home /home/openagent openagent \
    && mkdir -p /data \
    && chown 10001:10001 /data
COPY --from=builder /wheels /wheels
RUN set -eu; \
    wheel="$(find /wheels -maxdepth 1 -name 'openagent-*.whl' -print -quit)"; \
    test -n "$wheel"; \
    python -m pip install --no-cache-dir "${wheel}[media,security]"; \
    rm -rf /wheels

USER 10001:10001
WORKDIR /home/openagent
VOLUME ["/data"]
EXPOSE 8765
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3).read()" || exit 1

ENTRYPOINT ["python", "-m", "openagent"]
CMD ["--host", "0.0.0.0", "--port", "8765", "--data-dir", "/data"]
