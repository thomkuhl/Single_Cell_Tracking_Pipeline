# Single Cell Tracking Pipeline

This repository contains scripts and notes for running Cellpose-based segmentation, single-cell tracking and FRET analysis for HEK293T imaging experiments.
---

# 1. Overview

This repository contains the full single-cell analysis pipeline used for VPC trasfected HEK293T cells experiments.  
It includes all three major components of the workflow:

1. **Segmentation** — Cellpose-based segmentation with tuned parameters  
2. **Tracking & Labeling** — stable cell ID assignment across frames, handling merges/splits, generating labeled videos and CSV track files  
3. **FRET Analysis** — extraction of single-cell FRET traces  

All scripts used for segmentation, tracking, and FRET analysis are included in this repository.


---

# 2. Requirements

You need:

- Python 3 (for example 3.9 or 3.10).  
- A working Python environment (conda or venv).  
- For GPU acceleration: an NVIDIA GPU with CUDA drivers and a CUDA-enabled PyTorch installation.

Large image files (TIFF, ND2) are not required to be in the repository. Scripts, commands, and installation instructions will be provided in GitHub.

---

# 3. Installation

First create and activate a Python environment (example with conda):

```bash
conda create -n cellpose_env python=3.10
conda activate cellpose_env
```

Install the common Python packages:

```
pip install numpy scipy pandas matplotlib tifffile opencv-python scikit-image
```

---

# 3.1. Install Cellpose (CPU)

If you only want to run on CPU:
```
pip install cellpose
```

---

# 3.2. Install Cellpose (GPU, recommended)

GPU use requires a CUDA-enabled PyTorch.

**Step 1:** Install PyTorch with CUDA support. Follow the instructions on the PyTorch website and choose the command that matches your system:
```
https://pytorch.org/get-started/locally/
```

Example (only as a template; your command may differ):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
**Step 2:** Install Cellpose with GPU extras:
```
pip install "cellpose[gpu]"
```
**Step 3:** Test that Cellpose can see the GPU:
```
python -m cellpose --gpu_test
```
If the output shows “GPU activated”, the GPU installation is working. If it reports a fallback to CPU, check the PyTorch and CUDA versions.

---

# 4. Running Cellpose

Below is the standard command used for segmenting our HEK293T VPC FOVs.  
Change the directory path (`--dir`) to match the folder that contains your images.

```bash
python -m cellpose \
    --use_gpu \ #if available
    --verbose \ #enables detailed logging
    --dir ... \ #"path/to/frames/of/each/FOV"
    --pretrained_model cyto3 \  #Model used for these experiments
    --chan 0 \ #Use the first channel
    --diameter ... \ #Approx. cell diameter in pixels
    --cellprob_threshold ... \ #Controls how confident Cellpose must be that something is a cell (typical range of -10.0 to 10.0)
    --flow_threshold ... \ #Controls how well the flows must align for segmentation (typical range of 0.0 to 1.0)
    --min_size ... \ #Minimum allowed object size
    --norm_percentile ... \ #Normalization(example: 5 99.5)
    --save_tif \ #Save masks as TIFF
    --no_npy #Don’t save NumPy files, only TIFF(optional)
```

# 5. Suggested Project Layout

To keep the repository organized, the following layout is recommended:

```text
.
├── README.md
├── .gitignore
│
├── tracking/
│   ├── track_render_label_overlay.py #Contains all tracking and labeling scripts (Hungarian matching, velocity prediction, anti-merge logic, GIF/MP4 generation, etc.)
│
└── Single_cell_FRET/
    ├── Single_cell_fret.py #Contains all scripts for single-cell FRET extraction

```

**Raw imaging data (TIFF/ND2 files) will NOT be uploaded.**
