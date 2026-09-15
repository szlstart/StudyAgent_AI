FROM python:3.13-slim

ARG EVEROS_VERSION=1.2.3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EVEROS_ROOT=/data \
    EVEROS_API__HOST=0.0.0.0 \
    EVEROS_API__PORT=8000

RUN python -m pip install --no-cache-dir "everos==${EVEROS_VERSION}"

# everalgo-agent-memory 0.4.0 constructs DetectionResult with the 0.2.x
# signature. everalgo-boundary 0.3.0 added a required `should_wait` field and
# currently breaks EverOS 1.2.3 writes at runtime.
RUN python -m pip install --no-cache-dir "everalgo-boundary==0.2.0"

COPY --chmod=755 docker/everos/entrypoint.sh /usr/local/bin/everos-entrypoint

RUN useradd --create-home --uid 10001 everos \
    && mkdir -p /data \
    && chown -R everos:everos /data

USER everos
VOLUME ["/data"]
EXPOSE 8000

ENTRYPOINT ["everos-entrypoint"]
CMD ["everos", "server", "start"]
