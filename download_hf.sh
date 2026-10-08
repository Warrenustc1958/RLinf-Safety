#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_ROOT="/openbayes/input/input0"
DATASET_ROOT="/openbayes/input/input1"

# 使用国内 Hugging Face 镜像。
export HF_ENDPOINT="https://hf-mirror.com"

# 网络较慢时避免默认超时过短。
export HF_HUB_DOWNLOAD_TIMEOUT="600"
export HF_HUB_ETAG_TIMEOUT="60"

# 避免 hf-xet 绕开 HF_ENDPOINT；统一通过普通 HTTP 下载。
export HF_HUB_DISABLE_XET="1"

# 并发数。网络不稳定时可以改成 2 或 4。
WORKERS="${WORKERS:-8}"

echo "=== 检查目标目录是否可写 ==="
for directory in "$MODEL_ROOT" "$DATASET_ROOT"; do
    if [[ ! -d "$directory" ]]; then
        echo "错误：目录不存在：$directory" >&2
        exit 1
    fi

    test_file="$directory/.write_test_$$"
    if ! touch "$test_file" 2>/dev/null; then
        echo "错误：目录不可写：$directory" >&2
        echo "OpenBayes 的 input 目录可能是只读数据集挂载。" >&2
        exit 1
    fi
    rm -f "$test_file"
done

echo "=== 安装/升级 Hugging Face CLI ==="
python -m pip install "huggingface_hub==0.36.0"

if ! command -v hf >/dev/null 2>&1; then
    echo "错误：安装后仍找不到 hf 命令。" >&2
    exit 1
fi

echo "=== 磁盘空间 ==="
df -h "$MODEL_ROOT" "$DATASET_ROOT" || true

echo "=== 1/5 下载 Wan2.2-TI2V-5B ==="
hf download \
    Wan-AI/Wan2.2-TI2V-5B \
    --local-dir "$MODEL_ROOT/Wan2.2-TI2V-5B" \
    --max-workers "$WORKERS"

echo "=== 2/5 下载 pi05_libero_safety ==="
hf download \
    LIBERO-Safety/pi05_libero_safety \
    --local-dir "$MODEL_ROOT/pi05_libero_safety" \
    --max-workers "$WORKERS"

echo "=== 3/5 下载完整 LIBERO-Safety-Zip 数据集 ==="
hf download \
    Warrenustc1958/LIBERO-Safety-Zip \
    --repo-type dataset \
    --local-dir "$DATASET_ROOT/LIBERO-Safety-Zip" \
    --max-workers "$WORKERS"

echo "=== 4/5 仅下载 libero_safety/meta ==="
hf download \
    LIBERO-Safety/libero_safety \
    --repo-type dataset \
    --include "meta/**" \
    --local-dir "$DATASET_ROOT/libero_safety" \
    --max-workers "$WORKERS"

echo "=== 5/5 下载完整 libero_safety_assets ==="
hf download \
    LIBERO-Safety/libero_safety_assets \
    --repo-type dataset \
    --local-dir "$DATASET_ROOT/libero_safety_assets" \
    --max-workers "$WORKERS"

echo "=== 验证：libero_safety 下只能有 meta 和下载元数据目录 ==="
find "$DATASET_ROOT/libero_safety" \
    -mindepth 1 -maxdepth 1 \
    -printf '%f\n' | sort

if [[ -e "$DATASET_ROOT/libero_safety/data" ||
      -e "$DATASET_ROOT/libero_safety/videos" ]]; then
    echo "错误：检测到不应下载的 data 或 videos 目录。" >&2
    exit 1
fi

echo "=== 各目录占用 ==="
du -sh \
    "$MODEL_ROOT/Wan2.2-TI2V-5B" \
    "$MODEL_ROOT/pi05_libero_safety" \
    "$DATASET_ROOT/LIBERO-Safety-Zip" \
    "$DATASET_ROOT/libero_safety/meta" \
    "$DATASET_ROOT/libero_safety_assets"

echo "全部下载完成。"