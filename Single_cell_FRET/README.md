# Single Cell FRET Extraction

This repository provides a robust, track-aware single-cell FRET extraction pipeline designed for Cellpose-segmented timelapse data.  
The extractor uses stable track IDs, per-frame donor/acceptor images, Sauvola gating (optional), ring-based background subtraction, and bleed-through correction to produce clean, time-aligned single-cell FRET traces.

The script supports:

- Ring-based background subtraction  
- Optional Sauvola foreground gating  
- Gaussian smoothing  
- Bleed-through correction  
- Track-aware FRET computation  
- Per-track Excel sheets  
- Per-track PNG plots  
- Full parameter control via command-line arguments  

---

## 1. Folder Structure

Your data should look like:
```
FOV_XX/
├── Segmented/
│ ├── frame_000_cp_masks.tif
│ ├── frame_001_cp_masks.tif
│ └── ...
├── Donor/
│ ├── frame_000.tif
│ └── ...
├── Acceptor/
│ ├── frame_000.tif
│ └── ...
└── tracks_<name>.csv (output from the tracking pipeline)
```
## 2. Override parameters

Example: adjust Sauvola parameters + ring thickness:

```Single_cell_fret.py
--mask-dir /path/to/FOV_40/Segmented
--donor-dir /path/to/FOV_40/Donor
--acceptor-dir /path/to/FOV_40/Acceptor
--tracks-csv /path/to/FOV_40/tracks_<name>.csv
--sauvola-window 401
--sauvola-k 0.12
--ring-dilate 10
--ring-thick 3
```

Example: disable Sauvola gating:
```
python Single_cell_fret.py
--mask-dir /path/to/FOV_12/Segmented
--donor-dir /path/to/FOV_12/Donor
--acceptor-dir /path/to/FOV_12/Acceptor
--tracks-csv /path/to/FOV_12/tracks_<name>.csv
--no-sauvola
```

Show all parameters:
```
python Single_cell_fret.py --help
```



---

## 3. Outputs

The script generates:
```
fret_traces_<tag>.csv
fret_traces_<tag>.xlsx (optional Excel output)
fret_trace_plots_<tag>/ (per-track PNG traces)
tracks_ge<MIN_FRAMES>.csv (persistent track summary)
tracks_ge<MIN_FRAMES>.xlsx
```
Tracking outputs referenced by this pipeline:
```
tracks_<name>.csv (required input)
tracks_overlay_slow_<name>.mp4 (optional visualization)
overlays_<name>/ (per-frame tracking overlays)
```

Each FRET row includes:

- `track_id`  
- `frame`  
- `time_hr`  
- `donor_mean`, `acceptor_mean`  
- `donor_bg`, `acceptor_bg`  
- `donor_corr`, `acceptor_corr`  
- `fret_ratio`  

---

## 4. Parameters and Suggested Ranges (Compact)

**Track Filtering**
- `min-frames`: 20–80

**Timing**
- `frame-interval-min`: 5–15

**Ring Background**
- `ring-dilate`: 6–20  
- `ring-thick`: 1–5

**Foreground Gating**
- `use-sauvola`  
- `sigma-smooth`: 0–3  
- `sauvola-window`: 201–801  
- `sauvola-k`: 0.05–0.25  
- `erode-pixels`: 0–3

**Bleed-Through**
- `alpha`: 0.0–0.2  
- `beta`: 0.0–0.2

**Saving**
- `write-excel`  
- `save-plots`  
- `plot-tag`  

---

## 5. Recommended Presets

**Standard HEK293T p38 FRET movies**
- `ring-dilate 12`
- `ring-thick 2`
- `sauvola-window 551`
- `sauvola-k 0.15`
- `sigma-smooth 1.0`

**Noisy illumination**
- `sigma-smooth 2.0`
- `sauvola-window 801`
- `erode-pixels 2`

**Dim acceptor channel**
- `--no-sauvola`
- `ring-dilate 8`
- `ring-thick 3`

---

## 6. Citation

If you use this code in a publication, please cite this repository in the Methods.
