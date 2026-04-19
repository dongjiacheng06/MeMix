# Evaluation

This document describes the public MeMix evaluation setup for:

- 3D reconstruction
- camera pose estimation
- video depth estimation

## Datasets

Please follow the dataset download instructions from
[MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md)
and
[Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md)
to obtain:

- ScanNet
- TUM-dynamics
- Sintel
- Bonn
- KITTI
- Neural RGBD
- 7-Scenes

## Data Preparation

All prepare scripts read from `DATA_ROOT`, which defaults to `./data`.

If your datasets live elsewhere:

```bash
cd /path/to/MeMix
export DATA_ROOT=/path/to/your/data
```

Expected raw dataset roots under the default repo layout:

- `data/7scenes`
- `data/neural_rgbd`
- `data/tum`
- `data/scannetv2`
- `data/kitti`
- `data/bonn/rgbd_bonn_dataset`
- `data/sintel/training`

Prepare all public benchmarks with:

```bash
python datasets_preprocess/prepare_7scenes.py
python datasets_preprocess/prepare_nrgbd.py
python datasets_preprocess/prepare_tum.py
python datasets_preprocess/prepare_scannet.py
python datasets_preprocess/prepare_kitti.py
python datasets_preprocess/prepare_bonn.py
python datasets_preprocess/prepare_sintel.py
```

## Model Selection

All public eval entrypoints use the same `MODEL_NAMES` environment variable.

Examples:

```bash
CUDA_VISIBLE_DEVICES=0 bash eval/mv_recon/run.sh

MODEL_NAMES="cut_memix" CUDA_VISIBLE_DEVICES=0 bash eval/relpose/run_tum.sh

MODEL_NAMES="ttt ttt_memix" CUDA_VISIBLE_DEVICES=0 bash eval/video_depth/run_kitti.sh
```

Default six-model sweep:

```bash
MODEL_NAMES="cut ttt ttsa cut_memix ttt_memix ttsa_memix"
```

## Sequence Controls

For camera pose and video depth, the default run scripts use the paper short
setting. Use `EVAL_PROTOCOL=long` and `SEQ_LENGTHS` for long-sequence sweeps.
You can also override the frame window with:

- `START_FRAME`: first frame index
- `POSE_EVAL_STRIDE`: frame stride
- `MAX_VIEWS`: number of frames after striding

## 1. 3D Reconstruction

Datasets:

- 7-Scenes
- NRGBD

Run:

```bash
CUDA_VISIBLE_DEVICES=0 bash eval/mv_recon/run.sh
```

`MAX_VIEWS` can be either a single value or a whitespace-separated sweep:

```bash
MAX_VIEWS=300 KF_EVERY=2 CUDA_VISIBLE_DEVICES=0 bash eval/mv_recon/run.sh

MAX_VIEWS="300 400 500" KF_EVERY=2 \
CUDA_VISIBLE_DEVICES=0 bash eval/mv_recon/run.sh
```

For dense sampling, use `KF_EVERY=1` with the same `MAX_VIEWS` interface.

## 2. Camera Pose Estimation

Datasets:

- TUM and ScanNet for short and long settings
- Sintel for the short setting

Run:

```bash
CUDA_VISIBLE_DEVICES=0 bash eval/relpose/run_tum.sh
CUDA_VISIBLE_DEVICES=0 bash eval/relpose/run_scannet.sh
CUDA_VISIBLE_DEVICES=0 bash eval/relpose/run_sintel.sh
```

Examples:

```bash
# Short-sequence pose setting: TUM / ScanNet use 90 views.
MAX_VIEWS=90 CUDA_VISIBLE_DEVICES=0 bash eval/relpose/run_tum.sh

# Long-sequence pose sweep.
EVAL_PROTOCOL=long SEQ_LENGTHS="50 100 150 200 300 400 500 600 700 800 900 1000" \
CUDA_VISIBLE_DEVICES=0 bash eval/relpose/run_tum.sh
```

## 3. Video Depth Estimation

Datasets:

- KITTI
- Bonn
- Sintel

Run:

```bash
CUDA_VISIBLE_DEVICES=0 bash eval/video_depth/run_kitti.sh
CUDA_VISIBLE_DEVICES=0 bash eval/video_depth/run_bonn.sh
CUDA_VISIBLE_DEVICES=0 bash eval/video_depth/run_sintel.sh
```

Examples:

```bash
# Short-sequence depth setting: KITTI / Bonn use 110 views; Sintel uses 50.
MAX_VIEWS=110 CUDA_VISIBLE_DEVICES=0 bash eval/video_depth/run_kitti.sh

# Long-sequence depth sweep.
EVAL_PROTOCOL=long SEQ_LENGTHS="50 100 150 200 300 400 500 600 700 800 900 1000" \
CUDA_VISIBLE_DEVICES=0 bash eval/video_depth/run_sintel.sh
```

## Options

Shared:

- `MODEL_NAMES="cut ttt ttsa cut_memix ttt_memix ttsa_memix"`
- `NUM_PROCESSES=1`
- `PORT=29502`
- `WEIGHTS=/path/to/cut3r_512_dpt_4_64.pth`
- `DATA_ROOT=/path/to/data`

Runtime-sliced tasks (`relpose` and `video_depth`):

- `EVAL_PROTOCOL=short|long`
- `SEQ_LENGTHS="..."`
- `START_FRAME=<int>`
- `POSE_EVAL_STRIDE=<int>`
- `MAX_VIEWS=<int>`

`mv_recon` only:

- `DATASETS="7scenes NRGBD"`
- `MAX_VIEWS=300`
- `KF_EVERY=2`
