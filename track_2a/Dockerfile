FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hfcache PYTHONPATH=/app:/opt/SwissGov-RSD \
    OMP_NUM_THREADS=8
# Upstream SwissGov-RSD recognizer only, pinned to the exact tested revision.
# Sparse checkout keeps official gold data out of the image.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt
RUN git clone --filter=blob:none --no-checkout https://github.com/ZurichNLP/SwissGov-RSD.git /opt/SwissGov-RSD \
    && cd /opt/SwissGov-RSD \
    && git sparse-checkout init --cone \
    && git sparse-checkout set rsd \
    && git checkout 1807a42100e742ed03d337c54c4b9ea86995f565 \
    && test "$(git rev-parse HEAD)" = '1807a42100e742ed03d337c54c4b9ea86995f565'
COPY src/ /app/src/
COPY prompts/ /app/prompts/
COPY docs/ /app/docs/
COPY Makefile Dockerfile /app/
RUN mkdir -p /app/data/inputs /app/out/predictions /app/out/checkpoints /app/out/plans /opt/hfcache
CMD ["python", "-m", "src.submission", "run"]
