# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLAMA_HOME=/data \
    VLLAMA_HOST=0.0.0.0 \
    VLLAMA_PORT=11434

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip git curl \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# vllm first — large wheel, separate layer for cache efficiency
RUN pip install --upgrade pip && pip install vllm

WORKDIR /app
COPY . .
RUN pip install -e .

VOLUME ["/data", "/root/.cache/huggingface"]

EXPOSE 11434

CMD ["vllama", "serve"]
