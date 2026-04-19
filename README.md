<h2 align="center"><a href="https://dongjiacheng06.github.io/MeMix/">MeMix: Writing Less, Remembering More for Streaming 3D Reconstruction</a></h2>

<h5 align="center">

  [![Arxiv](https://img.shields.io/badge/Arxiv-2603.15330-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2603.15330)
  [![Home Page](https://img.shields.io/badge/Project-Website-33728E.svg)](https://dongjiacheng06.github.io/MeMix/)
  [![Code](https://img.shields.io/badge/Code-GitHub-181717.svg?logo=github)](https://github.com/dongjiacheng06/MeMix)

  [Jiacheng Dong](https://dongjiacheng06.github.io/)<sup>2*</sup>,
  [Huan Li](https://github.com/HuanLi0311)<sup>1*</sup>,
  [Sicheng Zhou](https://tzblog.tech/about/)<sup>2*</sup>,
  [Wenhao Hu](https://whhu7.github.io/)<sup>2</sup>,
  [Weili Xu](https://weili-0234.github.io/)<sup>2</sup>,
  [Yan Wang](https://yanwang202199.github.io/)<sup>1†</sup>

  <sup>1</sup>Institute for AI Industry Research, Tsinghua
  University
  <sup>2</sup>Zhejiang University

  <sup>*</sup>Equal contribution. <sup>†</sup>Corresponding
  author.

</h5>

<div align="center">TL;DR: Training-free selective memory updates for long-horizon recurrent streaming 3D reconstruction.</div>

https://github.com/user-attachments/assets/56b08162-c8b6-4251-9a01-038cd5f746b4

## Getting Started

### Installation

1. Clone MeMix.
```bash
git clone https://github.com/dongjiacheng06/MeMix
cd MeMix
```

2. Create the environment.
```bash
conda create -n memix python=3.11 cmake=3.14.0
conda activate memix
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# issues with pytorch dataloader, see https://github.com/pytorch/pytorch/issues/99625
conda install 'llvm-openmp<16'
```

3. Compile the RoPE CUDA kernels if you want the fast implementation:
```bash
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

### Download Checkpoints

CUT3R provides a checkpoint trained on 4-64 views:
[`cut3r_512_dpt_4_64.pth`](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link).

Download it with:
```bash
cd src
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ..
```

## Inference Demo

`demo.py` runs on a user-provided video file or image folder. Choose one public
variant from:

- `cut`
- `ttt`
- `ttsa`
- `cut_memix`
- `ttt_memix`
- `ttsa_memix`

Set `--seq_path` to your own local input and choose the variant with
`--model_variant`. If `--views` is not passed, the demo uses all decoded frames;
if it is passed, the demo uniformly samples that many views from the decoded
sequence after applying `--frame_interval`.

```bash
MODEL_VARIANT=cut_memix
CUDA_VISIBLE_DEVICES=0 python demo.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path /path/to/your_video.mp4 \
    --output_dir tmp/demo_run \
    --port 8080 \
    --model_variant ${MODEL_VARIANT} \
    --frame_interval 4 \
    --views 200 \
    --reset_interval 100 \
    --downsample_factor 20 \
    --vis_threshold 2.0
```

Demo outputs are written to `--output_dir`.

## Evaluation

Please refer to the [eval.md](eval/eval.md) for more details.

## Acknowledgements

Our code is built on top of the following open-source projects:

- [CUT3R](https://github.com/CUT3R/CUT3R)
- [TTT3R](https://github.com/Inception3D/TTT3R)
- [TTSA3R](https://github.com/anonus2357/ttsa3r)
- [Easi3R](https://github.com/Inception3D/Easi3R)
- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r)
- [Spann3R](https://github.com/HengyiWang/spann3r)
- [Viser](https://github.com/nerfstudio-project/viser)

We thank the authors for releasing their code.

## Citation

If you find our work useful, please cite:

```bibtex
@article{dong2026memix,
    title = {MeMix: Writing Less, Remembering More for Streaming 3D Reconstruction},
    author = {Dong, Jiacheng and Li, Huan and Zhou, Sicheng and Hu, Wenhao and Xu, Weili and Wang, Yan},
    journal = {arXiv preprint arXiv:2603.15330},
    year = {2026},
}
```
