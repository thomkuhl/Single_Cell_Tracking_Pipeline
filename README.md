# Single Cell Tracking Pipeline

This repository contains scripts and notes for running Cellpose-based segmentation and, later on, cell tracking and analysis for single-cell HEK293T (transfected with VPC construct) imaging experiments.
---

# 1. Overview

The current focus of this repository is the first stage of the pipeline: segmentation with Cellpose. For each field of view (FOV), Cellpose is run on the corresponding image folder and produces mask files (for example TIFF masks) that can then be used for tracking and downstream analysis.

Later, the repository can be extended with tracking scripts (stable cell IDs over time) and analysis scripts (FRET traces, spectral analysis, etc.).

---

# 2. Requirements

You need:

- Python 3 (for example 3.9 or 3.10).  
- A working Python environment (conda or venv).  
- For GPU acceleration: an NVIDIA GPU with CUDA drivers and a CUDA-enabled PyTorch installation.

Large image files (TIFF, ND2) are not required to be in the repository. Only scripts and small example outputs will be tracked in GitHub.

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
└── scripts/
    ├── track_cells.py           # Tracking and Labeling segmented cells script
    └── fret_analysis.py         # FRET Analysis




