# Installation for Robust OVAR

This document provides a complete installation procedure for this repository:

**Robust Zero-Shot Generalization for Open-Vocabulary Action Recognition via Task Arithmetic**

The project is based on Open-VCLIP and PySlowFast. The instructions below create
a dedicated Conda environment, install the dependencies used by Open-VCLIP,
build the local SlowFast package, and provide quick checks to verify that the
installation is usable for training, evaluation, model merging, and dataset
distance analysis.

---

## 1. Prerequisites

Before starting, make sure that the system provides:

* Conda or Miniconda;
* Python 3.8 support;
* CUDA-compatible NVIDIA drivers if GPU execution is required;
* GCC >= 4.9;
* Git;
* enough disk space for checkpoints, datasets, and external dependency clones.

The original setup used:

```text
Python 3.8.13
PyTorch 1.11.0
Torchvision 0.12.0
Torchaudio 0.11.0
CUDA toolkit 11.3
```

If your cluster or workstation uses a different CUDA version, adapt the PyTorch
installation command accordingly.

---

## 2. Create and Activate the Conda Environment

```bash
conda create -n ovar python=3.8.13 pip
conda activate ovar
```

---

## 3. Install PyTorch

For CUDA 11.3:

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

Check the installation:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

---

## 4. Install Python Dependencies

Install the general dependencies used by the repository:

```bash
pip install scipy
pip install pandas
pip install scikit-learn
pip install ftfy
pip install regex
pip install yacs
pip install pyyaml
pip install matplotlib
pip install termcolor
pip install tqdm
pip install simplejson
pip install psutil
pip install opencv-python
pip install tensorboard
pip install -U iopath
```

Install `fvcore`:

```bash
pip install 'git+https://github.com/facebookresearch/fvcore'
```

Install `av` from conda-forge:

```bash
pip install av

```

---

## 5. Install PyTorchVideo

Clone and install PyTorchVideo in editable mode:

```bash
mkdir -p /path/to/external/dependencies
cd /path/to/external/dependencies

git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip install -e .
```

---

## 6. Install FairScale
Install Detectron2

```bash
pip install 'git+https://github.com/facebookresearch/fairscale'
```
---

## 7. Install Detectron2

Clone and install Detectron2 in editable mode:

```bash
cd /path/to/external/dependencies
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Detectron2 can be sensitive to the installed PyTorch, CUDA, compiler, and Python
versions. If this editable install fails, install a Detectron2 build compatible
with your local PyTorch/CUDA setup.

---

## 8. Optional: Install OpenAI CLIP for Dataset Distance Analysis

The Open-VCLIP model code includes its own CLIP implementation under
`slowfast/models/clip`. However, the utilities in `dataset_distance/` import the
external `clip` package.

Install it if you plan to use dataset distance analysis:

```bash
pip install 'git+https://github.com/openai/CLIP.git'
```

---

## 9. Build the Local SlowFast Package

Go to the repository root:

```bash
cd /path/to/robust-ovar/repository
```

Expose the local `slowfast` package:

```bash
export PYTHONPATH=/path/to/robust-ovar/slowfast:$PYTHONPATH
```

Build the package in development mode:

```bash
python setup.py build develop
```

For future sessions, add the `PYTHONPATH` export to your shell startup file or
to the scripts you use to launch training and evaluation.

---

## 10. Download or Prepare Checkpoints

The project expects Open-VCLIP checkpoints to be available locally. Example
placeholder paths used by the scripts are:

```text
/path/to/openvclip/checkpoints/openvclip-b16/swa_2_22.pth
/path/to/openvclip/checkpoints/openvclip-l14/swa_2_22.pth
```

The experiment checkpoints are referenced in the repository README. After
downloading them, update the script variables such as:

```text
LOAD_CKPT_FILE
BASE_CKPT_FILE
SOURCE_CKPTS
CLIP_ORI_PATH
```

For ViT-B/16, use a ViT-B/16 Open-VCLIP checkpoint and a compatible CLIP
checkpoint, for example:

```text
/path/to/clip/cache/ViT-B-16.pt
```

For ViT-L/14, use ViT-L/14-compatible checkpoints and replace the config file:

```text
configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml
```

with:

```text
configs/Kinetics/TemporalCLIP_vitl14_8x16_STAdapter.yaml
```

---

## 11. Prepare Datasets

The training and evaluation scripts expect each dataset to follow this layout:

```text
/path/to/datasets/root/<dataset_name>/
  annotations/
  videos/
```

The scripts use the following dataset names and class-mapping files:

| Dataset | Script Dataset Name | Class Mapping |
| --- | --- | --- |
| HMDB51 | `hmdb51` | `annotations/hmdb51-index2cls.json` |
| UCF101 | `ucf101` | `annotations/ucf101-index2cls.json` |
| Kinetics-700 | `k700-2020` | `annotations/k700-2020-index2cls.json` |
| XD-Violence | `XD-Violence` | `annotations/XD-Violence-index2cls.json` |

Before running experiments, update the `DATA` variable in each script:

```text
DATA=/path/to/datasets/root
```

---

## 12. Verify the Installation

From the repository root:

```bash
cd /path/to/robust-ovar/repository
export PYTHONPATH=/path/to/robust-ovar/repository/slowfast:$PYTHONPATH
```

Run a basic import check:

```bash
python - <<'PY'
import torch
import torchvision
import av
import cv2
import pandas
import sklearn
import fvcore
import iopath
import slowfast

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("slowfast import: ok")
PY
```

Check that the main entrypoint is available:

```bash
python tools/run_net.py --help
```

Check the model merging script:

```bash
python model_merging.py --help
```

---

## 13. Running Project Scripts

After installation, edit the placeholder paths inside the scripts before
running them.

Training scripts:

```bash
bash scripts/training/ft-HMDB51-reg0.sh
bash scripts/training/ft-HMDB51-reg10.sh
bash scripts/training/ft-K700-reg0.sh
bash scripts/training/ft-K700-reg10.sh
bash scripts/training/ft-UCF101-reg0.sh
bash scripts/training/ft-UCF101-reg10.sh
bash scripts/training/ft-xdviolence-reg0.sh
bash scripts/training/ft-xdviolence-reg10.sh
```

Merging script:

```bash
bash scripts/merging/merging.sh
```

Evaluation scripts:

```bash
bash scripts/evaluation/eval-on-HMDB51.sh
bash scripts/evaluation/eval-on-K700.sh
bash scripts/evaluation/eval-on-UCF101.sh
bash scripts/evaluation/eval-on-XDV.sh
```

Merged-model evaluation scripts:

```bash
bash scripts/evaluation/merge-eval-on-HMDB51.sh
bash scripts/evaluation/merge-eval-on-K700.sh
bash scripts/evaluation/merge-eval-on-UCF101.sh
bash scripts/evaluation/merge-eval-on-XD-Violence.sh
```

---

## 14. Common Notes

If you get CUDA out-of-memory errors, reduce:

```text
TRAIN.BATCH_SIZE
TEST.BATCH_SIZE
```

ViT-L/14 usually requires smaller batch sizes than ViT-B/16.

If video decoding fails, verify that:

* `av` is installed in the active Conda environment;
* the video paths under `DATA.PATH_PREFIX` are correct;
* the annotation files point to existing videos;
* the decoding backend configured in the scripts matches the installed
  dependencies.

If imports fail, make sure the environment is active and `PYTHONPATH` includes:

```text
/path/to/robust-ovar/repository/slowfast
```

