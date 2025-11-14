Single Cell FRET Extraction

This repository provides a robust, track-aware single-cell FRET extraction pipeline designed for Cellpose-segmented timelapse data.
The extractor uses stable track IDs, per-frame donor/acceptor images, Sauvola gating (optional), ring-based background subtraction, and bleed-through correction to produce clean, time-aligned single-cell FRET traces.

The script supports:

Ring-based background subtraction

Optional Sauvola foreground gating

Gaussian smoothing

Bleed-through correction

Track-aware FRET computation

Per-track Excel sheets

Per-track PNG plots

Full parameter control via command-line arguments

1. Folder Structure

├── Segmented/
│   ├── frame_000_cp_masks.tif
│   ├── frame_001_cp_masks.tif
│   └── ...
├── Donor/
│   ├── frame_000.tif
│   └── ...
├── Acceptor/
│   ├── frame_000.tif
│   └── ...
└── tracks_v3.7_sticky_new.csv

This runs the FRET extractor with recommended defaults.

2. Override parameters

Example: adjust Sauvola parameters + ring thickness:

python fret_extract.py \
    --mask-dir /path/to/FOV_40/Segmented \
    --donor-dir /path/to/FOV_40/Donor \
    --acceptor-dir /path/to/FOV_40/Acceptor \
    --sauvola-window 401 \
    --sauvola-k 0.12 \
    --ring-dilate 10 \
    --ring-thick 3

Example: disable Sauvola gating:

python fret_extract.py \
    --mask-dir /path/to/FOV_12/Segmented \
    --donor-dir /path/to/FOV_12/Donor \
    --acceptor-dir /path/to/FOV_12/Acceptor \
    --no-sauvola

Show all parameters:

python fret_extract.py --help
