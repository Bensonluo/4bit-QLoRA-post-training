FROM python:3.11-slim-bookworm AS deps
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TUNESMITH_DISPLAY_ONLY=1 \
    MLFLOW_TRACKING_URI=sqlite:////app/outputs/display/mlflow.db \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY deploy/requirements-display.txt /tmp/requirements-display.txt
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements-display.txt
FROM deps AS runtime
RUN useradd --create-home --uid 10001 tunesmith
COPY --chown=tunesmith:tunesmith . /app
RUN mkdir -p /app/outputs && chown tunesmith:tunesmith /app/outputs
USER tunesmith
RUN python -m deploy.seed_display
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.baseUrlPath=qlora-dashboard", "--server.maxUploadSize=5", "--server.enableCORS=true", "--server.enableXsrfProtection=true", "--browser.gatherUsageStats=false"]
