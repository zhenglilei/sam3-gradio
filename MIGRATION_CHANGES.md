# SAM3 Gradio Migration Notes

Date: 2026-05-08

## Scope

This copy is the Zheng Qiyuan working copy and must run from:

- `/data/zhengqiyuan/sam3-gradio`

Do not modify the original source copy under `/data/zhenglilei` for this project.

## Runtime And Cache Directories

The demo now pins all runtime paths to Zheng Qiyuan owned directories:

- Gradio uploads: `/data/zhengqiyuan/sam3-gradio/.runtime/gradio`
- Python/system temp files: `/data/zhengqiyuan/sam3-gradio/.runtime/tmp`
- Video outputs: `/data/zhengqiyuan/sam3-gradio/.runtime/videos`
- Demo logs for managed runs: `/data/zhengqiyuan/sam3-gradio/.runtime/logs`
- Hugging Face cache: `/data/zhengqiyuan/.cache/huggingface`
- ModelScope cache: `/data/zhengqiyuan/.cache/modelscope`
- XDG cache root: `/data/zhengqiyuan/.cache`

This fixes the upload failure caused by Gradio defaulting to `/tmp/gradio`, which is owned by another user on this server.

## Model Path

The active image/video checkpoint remains:

- `/data/zhengqiyuan/sam3-gradio/models/sam3.pt`

It should point to Zheng Qiyuan's local ModelScope cache, not `/data/zhenglilei`.

## Polygon Mask Prompt

The image demo now has a `多边形Mask (Polygon)` mode and a `完成多边形对象` button.

Implementation strategy:

- Each positive polygon is rasterized into one binary mask.
- Each positive polygon is treated as one independent object.
- The demo runs SAM3's instance-interactive mask prompt once per polygon, then merges the object results for visualization.
- Negative polygons are rasterized as exclusion masks and subtracted from positive polygon masks.

This avoids changing the lower-level SAM3 geometry encoder, whose current image grounding path does not enable mask encoding and still assumes one mask prompt in parts of the code.

## Start Command

Use the project-local script:

```bash
cd /data/zhengqiyuan/sam3-gradio
./run_demo.sh
```

The script exports all runtime/cache environment variables before importing Gradio.
