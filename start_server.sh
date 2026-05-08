#!/bin/bash
set -euo pipefail

PROJECT_DIR=/data/zhengqiyuan/sam3-gradio
PYTHON_BIN=/data/zhengqiyuan/miniconda3/envs/sam3/bin/python
export TMPDIR="$PROJECT_DIR/.runtime/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export GRADIO_TEMP_DIR="$PROJECT_DIR/.runtime/gradio"
export XDG_CACHE_HOME=/data/zhengqiyuan/.cache
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export MODELSCOPE_CACHE="$XDG_CACHE_HOME/modelscope"

mkdir -p "$TMPDIR" "$GRADIO_TEMP_DIR" "$PROJECT_DIR/.runtime/videos" "$PROJECT_DIR/.runtime/logs" "$HF_HOME/hub" "$MODELSCOPE_CACHE"
cd "$PROJECT_DIR"

if [ ! -f "models/sam3.pt" ]; then
    echo "警告: 未找到模型文件 models/sam3.pt"
fi
if [ ! -f "assets/bpe_simple_vocab_16e6.txt.gz" ]; then
    echo "警告: 未找到BPE词汇表文件 assets/bpe_simple_vocab_16e6.txt.gz"
fi

echo "启动 SAM3 Gradio demo: http://0.0.0.0:7890"
exec "$PYTHON_BIN" sam3_gradio_demo.py
