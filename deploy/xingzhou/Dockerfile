FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-runtime.txt /opt/sam3/requirements-runtime.txt
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.7.0 torchvision==0.22.0 \
    && python -m pip install -r /opt/sam3/requirements-runtime.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/sam3/installed-requirements.txt

RUN mkdir -p /src && chown 10001:10001 /src
USER 10001:10001
WORKDIR /src
EXPOSE 8030
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8030/healthz', timeout=3)"
CMD ["/bin/sh", "/src/run.sh"]
