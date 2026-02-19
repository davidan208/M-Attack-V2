# Retrieval Pipeline (COCO)

This page documents how retrieval assets are built and used.

## What Most Users Should Do

If you already have these folders, you usually do **not** need to rebuild retrieval:

1. `resources/retrieved_embeddings`

Only rerun retrieval if you changed target images or want fresh retrieval results.
`resources/embeddings` is optional cache output and is generated automatically when retrieval runs.

## How Retrieval Is Used In Training

During adversarial generation (`generate_ad_sample_parallel.py` / `generate_ad_samples.py`):

1. `optim.use_retrieval=true`
2. `model.target_num=N`
3. the code loads `N-1` images from `resources/retrieved_embeddings/<target_id>/1.jpg ...`

Folder contract:

1. `resources/retrieved_embeddings/<target_image_stem>/1.jpg`
2. `resources/retrieved_embeddings/<target_image_stem>/2.jpg`
3. ...

If retrieval images are missing, the pipeline pads with the primary target image.

## Input Format Requirements

`retrieval.py` expects `ImageFolder`-style roots:

1. `root/<class_name>/<image files>`
2. target example: `resources/images/target_images/1/*.jpg`
3. COCO reference example: `resources/images/coco/train2014/*.jpg`

## Rebuild Retrieval (COCO)

The script now supports auto device selection.

```bash
# 1000-target setup
uv run python retrieval.py \
  --dataset coco \
  --target-images-dir resources/images/target_images \
  --device auto

# 100-target setup
uv run python retrieval.py \
  --dataset coco \
  --target-images-dir resources/images/target_images_100 \
  --device auto
```

## Outputs

1. `resources/retrieved_embeddings/<target_id>/1.jpg ... 5.jpg`
2. Optional cache files under `resources/embeddings/`:
   - `coco_embeddings.pt`
   - `target_embeddings.pt`

## Reproducibility Notes

To get the closest possible match with existing retrieval assets, keep:

1. the same target image set
2. the same COCO reference pool
3. the same software/model stack

Small differences can still happen if environment versions differ.

## Optional Verification

Quick checks after rebuilding:

```bash
# number of target IDs with retrieval folders
find resources/retrieved_embeddings -mindepth 1 -maxdepth 1 -type d | wc -l

# each ID folder should contain 5 retrieved images
python - <<'PY'
from pathlib import Path
root = Path("resources/retrieved_embeddings")
counts = [len(list(d.glob("*.jpg"))) for d in root.iterdir() if d.is_dir()]
print("dirs:", len(counts), "min:", min(counts), "max:", max(counts))
PY
```
