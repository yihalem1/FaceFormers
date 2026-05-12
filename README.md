# FaceFormers

Transformer-backbone 3D face reconstruction, built on top of the
[Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
pipeline. The standard ResNet-50 image encoder is replaced with a Vision
Transformer to shrink inference time while keeping reconstruction quality
comparable.

## What's in this repo

| Path | Purpose |
| --- | --- |
| [train.py](train.py) | Training loop (DDP-capable) for the face-reconstruction model |
| [test.py](test.py) | Inference script — reconstructs meshes from images in a folder and reports average inference time |
| [data_preparation.py](data_preparation.py) | Generates 68-point landmarks and skin masks, then writes the train/val file lists |
| [options/](options/) | `argparse` configs for base, train, and test runs |
| [data/](data/) | Dataset loaders (`flist`, image folder, base/template) |
| [util/preprocess.py](util/preprocess.py) | Face alignment, cropping, 5-point/68-point landmark utilities |
| [util/detect_lm68.py](util/detect_lm68.py) | 68-landmark detector wrapper (uses `checkpoints/lm_model/68lm_detector.pb`) |
| [util/skin_mask.py](util/skin_mask.py) | Skin-attention mask generation used as a training signal |
| [util/load_mats.py](util/load_mats.py) | Loads the Basel Face Model (BFM) parameters |
| [util/nvdiffrast.py](util/nvdiffrast.py) | Differentiable mesh renderer used in the photometric loss |
| [util/visualizer.py](util/visualizer.py) | TensorBoard / HTML visualizer for losses and rendered samples |
| [BFM/BFM_front_idx.mat](BFM/BFM_front_idx.mat) | Front-face vertex indices for the BFM mesh |
| [environment.yml](environment.yml) | Conda environment specification |

> **Heads up:** the `models/` package is imported by `train.py` and `test.py`
> (`from models import create_model`) but is **not bundled in this repo**. The
> contract is the same as in the upstream Deep3DFaceRecon_pytorch project —
> drop your transformer-backbone face-recon model definition into a
> `models/` directory exposing `create_model(opt)` and a
> `<model>_model.py` that implements `set_input`, `optimize_parameters`,
> `test`, `get_current_visuals`, `save_mesh`, and `save_coeff`.

## Setup

```bash
# 1. Clone
git clone https://github.com/yihalem1/FaceFormers.git
cd FaceFormers

# 2. Create the conda env (Python 3.6 / PyTorch 1.6 / CUDA-capable GPU recommended)
conda env create -f environment.yml
conda activate deep3d_pytorch

# 3. Download the Basel Face Model (BFM) into ./BFM/
#    Follow the instructions from the upstream Deep3DFaceRecon_pytorch repo:
#    https://github.com/sicxu/Deep3DFaceRecon_pytorch#prepare-prerequisite-models

# 4. Download the 68-landmark detector to ./checkpoints/lm_model/68lm_detector.pb
#    (also documented in the upstream repo)
```

## Data preparation

`data_preparation.py` walks one or more image folders, detects landmarks,
generates skin masks, and writes the train/val file lists.

```bash
python data_preparation.py \
    --data_root datasets \
    --img_folder ffhq celeba \
    --mode train
```

It expects each folder under `datasets/` to contain raw `.jpg`/`.png` images
and will populate `landmarks/` and `mask/` subdirectories alongside them.

## Training

```bash
python train.py \
    --name faceformer_run1 \
    --gpu_ids 0 \
    --batch_size 32 \
    --n_epochs 20 \
    --lr 1e-4
```

Useful flags (defaults shown):
- `--use_ddp True` — distributed data parallel across multiple GPUs
- `--ddp_port 12355`
- `--save_epoch_freq 1` — checkpoint cadence
- `--display_freq 1000` / `--evaluation_freq 5000`
- `--continue_train` — resume from the latest checkpoint in `checkpoints/<name>/`

See [options/train_options.py](options/train_options.py) and
[options/base_options.py](options/base_options.py) for the full list.

## Inference

Place test images and detection text files like this:

```
examples/
    img001.jpg
    img002.jpg
    detections/
        img001.txt   # 5 facial landmarks, one (x y) pair per line
        img002.txt
```

Then run:

```bash
python test.py \
    --name faceformer_run1 \
    --epoch latest \
    --img_folder examples
```

For each image the script saves:
- a reconstructed mesh as `.obj`
- predicted coefficients as `.mat`
- a side-by-side rendered visualization

It also prints the average inference time per image — the headline metric
this project optimizes.

## Why a transformer backbone?

The upstream model uses ResNet-50 to regress BFM coefficients from an
aligned 224×224 face crop. Substituting a Vision Transformer:

- preserves reconstruction quality (the regression head, BFM rendering loss,
  and skin-mask supervision are unchanged), and
- reduces per-image inference latency for real-time-ish use cases.

If you reproduce the experiments, the easiest comparison knob is to swap the
backbone in your `models/facerecon_model.py` between `resnet50` and the
transformer variant and rerun `test.py` on a fixed `examples/` set — the
average inference time printed at the end is the comparison number.

## Acknowledgements

This project builds directly on
[Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
by Sicheng Xu et al., which itself is a PyTorch reimplementation of
*Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From
Single Image to Image Set* (Deng et al., CVPRW 2019). All credit for the
underlying pipeline, BFM integration, and differentiable rendering belongs
to those authors.

## License

No license is currently attached to this repository. Until one is added,
default copyright applies — open an issue if you'd like to discuss usage,
or pick a permissive license (MIT / Apache-2.0) before publishing
derivatives.
