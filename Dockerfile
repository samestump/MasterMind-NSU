FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    HOME=/tmp \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir uv==0.13.7 \
    && uv pip install --system --no-cache -r requirements.lock \
    && pip uninstall -y uv
COPY harness ./harness
COPY scripts/estimator.py ./scripts/estimator.py
COPY run_agent.py ./
# Competition profile: the judge runs only --tasks-dir; these are the frozen defaults.
COPY competition ./competition
ENV TOKEN_PREDICTION_PROFILE=/app/competition/profile.json
ENTRYPOINT ["python", "/app/run_agent.py"]
