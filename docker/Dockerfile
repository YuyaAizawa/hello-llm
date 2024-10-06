FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    ssh \
    git \
    gosu \
    python3 python3-pip \
    python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip \
    && pip install --no-cache-dir \
    numpy \
    torch

WORKDIR /workspace/hello-llm
ENV FORCE_CUDA="1"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
