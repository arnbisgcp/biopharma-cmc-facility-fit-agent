FROM python:3.13-slim

RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --create-home appuser

WORKDIR /app
COPY . /app
RUN mkdir -p /app/data && chown -R appuser:appgroup /app

USER appuser
ENV PORT=8080
ENV FACILITY_FIT_DB_PATH=/app/data/facility_fit_memory.sqlite3
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')"

CMD ["python3", "main.py"]
