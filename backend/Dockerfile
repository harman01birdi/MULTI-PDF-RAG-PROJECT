FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf_cache

WORKDIR /app

RUN pip install --no-cache-dir "torch==2.14.1+cpu" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# dependencies before code, so editing code doesn't reinstall everything
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# run as a non-root user; /data is where the volume mounts
RUN useradd -m appuser && mkdir -p /data && chown -R appuser /app /data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

