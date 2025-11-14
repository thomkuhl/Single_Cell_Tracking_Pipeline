#!/usr/bin/env python
"""
track_render_label_overlay.py

Stable ID tracking for Cellpose masks:
  - Pre-assign "prior owner" when overlap is strong (LOCK step) to avoid ID flips.
  - Hungarian matching with IoU-dominant cost + constant-velocity prediction.
  - Anti-takeover prior & revive for brief disappearances.
  - Optional per-frame overlays + MP4 movie + track CSV.

Requires:
    pip install numpy tifffile opencv-python scipy pandas scikit-image
"""

import os
import re
import cv2
import tifffile
import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
import argparse


# ------------------------ Helpers ------------------------

def ensure_2d_bool(arr):
    a = np.asarray(arr)
    if a.ndim > 2:
        if a.ndim == 3 and a.shape[-1] in (3, 4):
            a = np.any(a[..., :3] != 0, axis=-1)
        else:
            a = np.squeeze(a)
            if a.ndim == 3:
                a = np.any(a, axis=0)
    return a.astype(bool)


def normalize_label_mask(mask_raw):
    m = np.asarray(mask_raw)
    if m.ndim > 2:
        m = np.squeeze(m)
        if m.ndim == 3 and m.shape[-1] in (3, 4):
            m = np.any(m[..., :3] != 0, axis=-1).astype(np.int32)
        elif m.ndim == 3:
            m = m[0].astype(np.int32)
        else:
            m = m.astype(np.int32)
    else:
        m = m.astype(np.int32)
    return m


def align_shape_like(reference_hw, target):
    refH, refW = reference_hw
    t = np.asarray(target)
    if t.shape == (refH, refW):
        return t
    if t.shape == (refW, refH):
        return t.T
    raise ValueError(f"Mask size mismatch: ref={reference_hw}, got={t.shape}")


def objects_from_mask(mask, min_area):
    objs = []
    for lbl in np.unique(mask):
        if lbl == 0:
            continue
        ys, xs = np.where(mask == lbl)
        area = ys.size
        if area < min_area:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        objs.append({"label": int(lbl), "cx": cx, "cy": cy, "area": int(area)})
    return objs


def build_label_masks(mask, objs):
    return {o["label"]: (mask == o["label"]) for o in objs}


def overlay_ids(base_img, id_img):
    if base_img.ndim == 2:
        denom = float(base_img.max()) or 1.0
        base8 = (255 * (base_img.astype(np.float32) / denom)).astype(np.uint8)
        rgb = cv2.cvtColor(base8, cv2.COLOR_GRAY2BGR)
    else:
        rgb = base_img[..., :3].copy()
        if rgb.dtype != np.uint8:
            mx = float(rgb.max()) or 1.0
            rgb = (255 * (rgb.astype(np.float32) / mx)).astype(np.uint8)
    out = rgb.copy()
    tids = [t for t in np.unique(id_img) if t != 0]
    for tid in tids:
        m = (id_img == tid)
        color = (int(37 * tid) % 255, int(97 * tid) % 255, int(173 * tid) % 255)
        out[m] = (0.7 * out[m] + 0.3 * np.array(color)).astype(out.dtype)
        ys, xs = np.nonzero(m)
        if xs.size:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(
                out,
                str(int(tid)),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return out


# ---------- Mask discovery (robust) ----------

def _natsort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_masks(root):
    pats = ("_cp_masks.tif", "_cp_masks.tiff")  # any case
    hits = []
    for fn in os.listdir(root):
        fcl = fn.casefold()
        if fcl.endswith(pats):
            hits.append(os.path.join(root, fn))
    hits.sort(key=_natsort_key)
    return hits


def stem_from_mask_path(mpath):
    # strip the trailing _cp_masks(.tif[f]) (case-insensitive)
    base = os.path.basename(mpath)
    return re.sub(r"(?i)_cp_masks(?:\.tif{1,2})?$", "", base)


# ------------------------ Core tracking function ------------------------

def track_and_render(
    mask_dir,
    raw_dir=None,
    name="v3.7_sticky_new",
    # motion / cost
    search_range=70.0,
    alpha=0.25,
    beta=0.75,
    gamma=0.05,
    max_cost=1.8,
    # memory
    memory=10,
    revive_gap=60,
    # morphology and base gating
    dilate_radius=11,
    min_area=50,
    # anti-takeover & revive strictness
    enforce_mutual_iou=True,
    mutual_iou_tol=1e-6,
    revive_iou_min=0.08,
    no_takeover_recent=8,
    # hard lock
    lock_iou=0.04,
    lock_dist=60.0,
    # rendering / export
    erode_px=0,
    fps=2.0,
    repeat=3,
    write_track_tiffs=False,
):
    """
    Run sticky tracking on a folder of Cellpose masks and optionally render overlays + MP4.

    Parameters mostly mirror the original CONFIG block, now passed as arguments.
    """

    if raw_dir is None:
        raw_dir = mask_dir

    # Outputs
    out_csv = os.path.join(mask_dir, f"tracks_{name}.csv")
    out_mp4 = os.path.join(mask_dir, f"tracks_overlay_slow_{name}.mp4")
    out_dir = os.path.join(mask_dir, f"overlays_{name}")
    tiff_dir = os.path.join(mask_dir, f"tracks_tiff_{name}")

    os.makedirs(out_dir, exist_ok=True)
    if write_track_tiffs:
        os.makedirs(tiff_dir, exist_ok=True)

    # Discover masks
    mask_files = find_masks(mask_dir)
    print(f"[INFO] Found {len(mask_files)} mask files in {mask_dir}")
    if len(mask_files) >= 3:
        print("[INFO] Last 3 masks:", [os.path.basename(p) for p in mask_files[-3:]])
    if not mask_files:
        raise FileNotFoundError(
            "No '*_cp_masks.tif' or '*_cp_masks.tiff' files in mask_dir"
        )

    tracks = {}  # tid -> dict(last_frame,cx,cy,area,mask,label,vx,vy)
    next_tid = 1
    rows = []

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    se_dilate = disk(dilate_radius) if dilate_radius > 0 else None

    ref_hw = None
    h0 = w0 = None

    seen_bases = []  # mask stems we attempted
    wrote_for = []   # mask stems that produced a PNG
    fail_reason = {}  # stem -> reason string

    for f_idx, mpath in enumerate(mask_files):
        stem = stem_from_mask_path(mpath)
        seen_bases.append(stem)
        try:
            # ---- Load & normalize mask ----
            raw_mask = tifffile.imread(mpath)
            mask = normalize_label_mask(raw_mask)
            if ref_hw is None:
                ref_hw = mask.shape
            mask = align_shape_like(ref_hw, mask).astype(np.int32)

            # Detections this frame
            objs = objects_from_mask(mask, min_area=min_area)
            lbl_masks = build_label_masks(mask, objs)

            # ---- Recent tracks (for prior ownership) ----
            recent_tids = [
                tid
                for tid, st in tracks.items()
                if 1 <= (f_idx - st["last_frame"]) <= no_takeover_recent
            ]

            # Map detection -> (best_tid, best_iou, d_pred) among recent owners
            prior_owner = {}
            for o in objs:
                cur_mask = ensure_2d_bool(lbl_masks[o["label"]])
                cur_mask = align_shape_like(ref_hw, cur_mask)
                best_tid, best_iou, best_d = None, 0.0, 1e9
                for tid in recent_tids:
                    st = tracks[tid]
                    prev_mask = ensure_2d_bool(st["mask"])
                    prev_mask = align_shape_like(ref_hw, prev_mask)
                    prev_mask_d = (
                        binary_dilation(prev_mask, structure=se_dilate)
                        if se_dilate is not None
                        else prev_mask
                    )
                    inter = np.logical_and(prev_mask_d, cur_mask).sum()
                    if inter == 0:
                        continue
                    union = prev_mask_d.sum() + cur_mask.sum() - inter
                    iou = float(inter) / float(union)
                    vx = st.get("vx", 0.0)
                    vy = st.get("vy", 0.0)
                    px, py = st["cx"] + vx, st["cy"] + vy
                    d_pred = float(np.hypot(o["cx"] - px, o["cy"] - py))
                    if iou > best_iou or (
                        abs(iou - best_iou) < 1e-9 and d_pred < best_d
                    ):
                        best_tid, best_iou, best_d = tid, iou, d_pred
                prior_owner[o["label"]] = (best_tid, best_iou, best_d)

            # ---- Active tracks (within memory) ----
            active_tids = [
                tid for tid, st in tracks.items() if f_idx - st["last_frame"] <= memory
            ]

            # ================== PRE-ASSIGN (HARD LOCK) ==================
            assigned_tracks = set()
            assigned_objs = set()
            lock_candidates = []
            for j, o in enumerate(objs):
                tid, iou, d_pred = prior_owner.get(o["label"], (None, 0.0, 1e9))
                if (
                    tid is not None
                    and iou >= lock_iou
                    and d_pred <= lock_dist
                ):
                    lock_candidates.append((iou, -d_pred, tid, j))  # sort: high IoU

            lock_candidates.sort(reverse=True)

            for iou, neg_d, tid, j in lock_candidates:
                if tid in assigned_tracks or j in assigned_objs:
                    continue
                o = objs[j]
                st = tracks[tid]
                cur_mask = ensure_2d_bool(lbl_masks[o["label"]])
                cur_mask = align_shape_like(ref_hw, cur_mask)
                vx_new = (o["cx"] - st["cx"])
                vy_new = (o["cy"] - st["cy"])
                tracks[tid] = {
                    "last_frame": f_idx,
                    "cx": o["cx"],
                    "cy": o["cy"],
                    "area": o["area"],
                    "mask": cur_mask.copy(),
                    "label": int(o["label"]),
                    "vx": 0.5 * vx_new + 0.5 * st.get("vx", 0.0),
                    "vy": 0.5 * vy_new + 0.5 * st.get("vy", 0.0),
                }
                rows.append(
                    {
                        "frame": f_idx,
                        "track_id": tid,
                        "cx": o["cx"],
                        "cy": o["cy"],
                        "area": o["area"],
                        "label": int(o["label"]),
                    }
                )
                assigned_tracks.add(tid)
                assigned_objs.add(j)

            # ---- Build cost matrix for remaining (Hungarian) ----
            rem_active = [tid for tid in active_tids if tid not in assigned_tracks]
            rem_obj_idxs = [j for j in range(len(objs)) if j not in assigned_objs]

            iou_mat = None
            if rem_active and rem_obj_idxs:
                nT, nO = len(rem_active), len(rem_obj_idxs)
                cost = np.full((nT, nO), 1e6, dtype=np.float32)
                iou_mat = np.zeros((nT, nO), dtype=np.float32)

                for i, tid in enumerate(rem_active):
                    st = tracks[tid]
                    prev_mask = ensure_2d_bool(st["mask"])
                    prev_mask = align_shape_like(ref_hw, prev_mask)
                    prev_mask_d = (
                        binary_dilation(prev_mask, structure=se_dilate)
                        if se_dilate is not None
                        else prev_mask
                    )

                    vx = st.get("vx", 0.0)
                    vy = st.get("vy", 0.0)
                    px, py = st["cx"] + vx, st["cy"] + vy

                    for jj, j in enumerate(rem_obj_idxs):
                        o = objs[j]
                        d_pred = np.hypot(o["cx"] - px, o["cy"] - py)
                        if d_pred > 3 * search_range:
                            continue
                        cur_mask = ensure_2d_bool(lbl_masks[o["label"]])
                        cur_mask = align_shape_like(ref_hw, cur_mask)
                        inter = np.logical_and(prev_mask_d, cur_mask).sum()
                        if inter == 0:
                            iou_d = 0.0
                        else:
                            union = prev_mask_d.sum() + cur_mask.sum() - inter
                            iou_d = float(inter) / float(union)

                        area_ratio = max(o["area"], 1) / max(st["area"], 1)
                        area_pen = abs(np.log(area_ratio))

                        if iou_d <= 0 and d_pred > search_range:
                            continue

                        jump_pen = min(1.0, d_pred / (2 * search_range))
                        c = (
                            alpha * (d_pred / max(1e-6, search_range)) ** 2
                            + beta * (1.0 - iou_d)
                            + gamma * area_pen
                            + 0.12 * jump_pen
                        )
                        cost[i, jj] = c
                        iou_mat[i, jj] = iou_d

                if np.isfinite(cost).any():
                    r, cidx = linear_sum_assignment(cost)
                    for i, jj in zip(r, cidx):
                        if cost[i, jj] <= max_cost:
                            tid = rem_active[i]
                            j = rem_obj_idxs[jj]
                            o = objs[j]

                            prior_tid, prior_iou, _ = prior_owner.get(
                                o["label"], (None, 0.0, 1e9)
                            )
                            if (
                                prior_tid is not None
                                and prior_iou >= revive_iou_min
                                and prior_tid != tid
                            ):
                                continue

                            if enforce_mutual_iou and iou_mat is not None and iou_mat[i, jj] > 0:
                                row_best = iou_mat[i, jj] >= (
                                    iou_mat[i, :].max() - mutual_iou_tol
                                )
                                col_best = iou_mat[i, jj] >= (
                                    iou_mat[:, jj].max() - mutual_iou_tol
                                )
                                if not (row_best and col_best):
                                    continue

                            st = tracks[tid]
                            cur_mask = ensure_2d_bool(lbl_masks[o["label"]])
                            cur_mask = align_shape_like(ref_hw, cur_mask)
                            vx_new = (o["cx"] - st["cx"])
                            vy_new = (o["cy"] - st["cy"])
                            tracks[tid] = {
                                "last_frame": f_idx,
                                "cx": o["cx"],
                                "cy": o["cy"],
                                "area": o["area"],
                                "mask": cur_mask.copy(),
                                "label": int(o["label"]),
                                "vx": 0.5 * vx_new + 0.5 * st.get("vx", 0.0),
                                "vy": 0.5 * vy_new + 0.5 * st.get("vy", 0.0),
                            }
                            rows.append(
                                {
                                    "frame": f_idx,
                                    "track_id": tid,
                                    "cx": o["cx"],
                                    "cy": o["cy"],
                                    "area": o["area"],
                                    "label": int(o["label"]),
                                }
                            )
                            assigned_tracks.add(tid)
                            assigned_objs.add(j)

            # ---- Try to reuse IDs for leftover detections (revive / reuse) ----
            def try_attach_to_existing(o, j):
                candidates = []
                cur_mask = ensure_2d_bool(lbl_masks[o["label"]])
                cur_mask = align_shape_like(ref_hw, cur_mask)

                # 1) Attach to active tracks with good IoU
                for tid in [t for t in active_tids if t not in assigned_tracks]:
                    st = tracks[tid]
                    vx = st.get("vx", 0.0)
                    vy = st.get("vy", 0.0)
                    px, py = st["cx"] + vx, st["cy"] + vy
                    d_pred = np.hypot(o["cx"] - px, o["cy"] - py)
                    if d_pred > 3 * search_range:
                        continue
                    prev_mask = ensure_2d_bool(st["mask"])
                    prev_mask = align_shape_like(ref_hw, prev_mask)
                    prev_mask_d = (
                        binary_dilation(prev_mask, structure=se_dilate)
                        if se_dilate is not None
                        else prev_mask
                    )
                    inter = np.logical_and(prev_mask_d, cur_mask).sum()
                    union = prev_mask_d.sum() + cur_mask.sum() - inter
                    iou_d = inter / union if union > 0 else 0.0
                    if iou_d < revive_iou_min:
                        continue
                    area_pen = abs(
                        np.log(max(o["area"], 1) / max(st["area"], 1))
                    )
                    cost_val = (
                        alpha * (d_pred / max(1e-6, search_range)) ** 2
                        + beta * (1.0 - iou_d)
                        + gamma * area_pen
                    )
                    candidates.append((1.0 - iou_d, cost_val, 0, tid))

                # 2) Revive ended tracks within revive_gap
                for tid, st in tracks.items():
                    if tid in assigned_tracks:
                        continue
                    gap = f_idx - st["last_frame"]
                    if gap <= memory or gap == 0 or gap > revive_gap:
                        continue
                    vx = st.get("vx", 0.0)
                    vy = st.get("vy", 0.0)
                    px, py = st["cx"] + vx * gap, st["cy"] + vy * gap
                    d_pred = np.hypot(o["cx"] - px, o["cy"] - py)
                    if d_pred > 2 * search_range:
                        continue
                    prev_mask = ensure_2d_bool(st["mask"])
                    prev_mask = align_shape_like(ref_hw, prev_mask)
                    prev_mask_d = (
                        binary_dilation(prev_mask, structure=se_dilate)
                        if se_dilate is not None
                        else prev_mask
                    )
                    inter = np.logical_and(prev_mask_d, cur_mask).sum()
                    union = prev_mask_d.sum() + cur_mask.sum() - inter
                    iou_d = inter / union if union > 0 else 0.0
                    if iou_d < revive_iou_min:
                        continue
                    area_pen = abs(
                        np.log(max(o["area"], 1) / max(st["area"], 1))
                    )
                    cost_val = (
                        alpha * (d_pred / max(1e-6, search_range)) ** 2
                        + beta * (1.0 - iou_d)
                        + gamma * area_pen
                        + 0.05 * gap
                    )
                    candidates.append((1.0 - iou_d, cost_val, gap, tid))

                if not candidates:
                    return None
                candidates.sort(key=lambda x: (x[0], x[1], x[2]))
                best = candidates[0]
                if best[1] <= max_cost:
                    return best[3]
                return None

            for j, o in enumerate(objs):
                if j in assigned_objs:
                    continue
                reuse_tid = try_attach_to_existing(o, j)
                if reuse_tid is not None:
                    st = tracks[reuse_tid]
                    cur_mask = ensure_2d_bool(lbl_masks[o["label"]])
                    cur_mask = align_shape_like(ref_hw, cur_mask)
                    vx_new = (o["cx"] - st["cx"])
                    vy_new = (o["cy"] - st["cy"])
                    tracks[reuse_tid] = {
                        "last_frame": f_idx,
                        "cx": o["cx"],
                        "cy": o["cy"],
                        "area": o["area"],
                        "mask": cur_mask.copy(),
                        "label": int(o["label"]),
                        "vx": 0.5 * vx_new + 0.5 * st.get("vx", 0.0),
                        "vy": 0.5 * vy_new + 0.5 * st.get("vy", 0.0),
                    }
                    rows.append(
                        {
                            "frame": f_idx,
                            "track_id": reuse_tid,
                            "cx": o["cx"],
                            "cy": o["cy"],
                            "area": o["area"],
                            "label": int(o["label"]),
                        }
                    )
                    assigned_tracks.add(reuse_tid)
                    assigned_objs.add(j)
                else:
                    tid = next_tid
                    next_tid += 1
                    cur_mask = ensure_2d_bool(lbl_masks[o["label"]])
                    cur_mask = align_shape_like(ref_hw, cur_mask)
                    tracks[tid] = {
                        "last_frame": f_idx,
                        "cx": o["cx"],
                        "cy": o["cy"],
                        "area": o["area"],
                        "mask": cur_mask.copy(),
                        "label": int(o["label"]),
                        "vx": 0.0,
                        "vy": 0.0,
                    }
                    rows.append(
                        {
                            "frame": f_idx,
                            "track_id": tid,
                            "cx": o["cx"],
                            "cy": o["cy"],
                            "area": o["area"],
                            "label": int(o["label"]),
                        }
                    )
                    assigned_tracks.add(tid)
                    assigned_objs.add(j)

            # ---- Build per-pixel ID image (paint by matched labels) ----
            id_img = np.zeros_like(mask, dtype=np.int32)
            for tid, st in tracks.items():
                if st["last_frame"] == f_idx:
                    lbl = st.get("label", None)
                    if lbl is not None:
                        id_img[mask == lbl] = tid

            # ---- Base image for overlay (raw if available) ----
            raw_path = None
            for ext in (".tif", ".tiff"):
                candidate = os.path.join(raw_dir, stem + ext)
                if os.path.exists(candidate):
                    raw_path = candidate
                    break

            if raw_path is not None:
                try:
                    raw = tifffile.imread(raw_path)
                    base_img = raw if raw.ndim == 2 else raw[..., 0]
                except Exception as e:
                    print(
                        f"[WARN] Could not read raw image {os.path.basename(raw_path)} -> {e}"
                    )
                    base_img = (mask > 0).astype(np.uint8) * 200
            else:
                base_img = (mask > 0).astype(np.uint8) * 200

            frame_vis = overlay_ids(base_img, id_img)

            # ---- Video writer init & size consistency ----
            if writer is None:
                h0, w0 = frame_vis.shape[:2]
                writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w0, h0))
            if frame_vis.shape[:2] != (h0, w0):
                frame_vis = cv2.resize(
                    frame_vis, (w0, h0), interpolation=cv2.INTER_NEAREST
                )

            # ---- Write overlay named by stem ----
            overlay_path = os.path.join(out_dir, f"{stem}_overlay.png")
            ok = cv2.imwrite(overlay_path, frame_vis)
            if not ok:
                fail_reason[stem] = "cv2.imwrite returned False"

            for _ in range(repeat):
                writer.write(frame_vis)

            if write_track_tiffs:
                tifffile.imwrite(
                    os.path.join(tiff_dir, f"{stem}_tracks.tif"),
                    id_img.astype(np.int32),
                )

            wrote_for.append(stem)

        except Exception as e:
            fail_reason[stem] = f"Exception: {type(e).__name__}: {e}"
            print(f"[WARN] Skipping mask {os.path.basename(mpath)} -> {e}")
            continue

    # finalize
    if writer is not None:
        writer.release()

    # CSV
    try:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    except PermissionError:
        base, ext = os.path.splitext(out_csv)
        alt = base + "_run.csv"
        pd.DataFrame(rows).to_csv(alt, index=False)
        print(
            f"[WARN] Could not write {out_csv} (in use). Wrote {alt} instead.)"
        )

    # End-of-run sanity + diff
    seen_set = set(seen_bases)
    wrote_set = set(wrote_for)
    missing = sorted(list(seen_set - wrote_set), key=_natsort_key)

    print(f"\n[INFO] Masks discovered : {len(seen_bases)}")
    print(f"[INFO] Overlays written: {len(wrote_for)} (expected {len(seen_bases)})")
    if missing:
        print("[INFO] Missing overlays for stems (and reason if known):")
        for s in missing[:50]:  # print up to 50
            print("   -", s, "=>", fail_reason.get(s, "unknown"))
        if len(missing) > 50:
            print(f"   ... and {len(missing) - 50} more")
    print("\nWrote:")
    print(" ", out_csv)
    print(" ", out_mp4)
    print(" ", out_dir)
    if write_track_tiffs:
        print(" ", tiff_dir)
    print("Done.")

    return out_csv, out_mp4, out_dir, (tiff_dir if write_track_tiffs else None)


# ------------------------ CLI ------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="Stable ID tracking for Cellpose masks with sticky matching."
    )
    p.add_argument(
        "--mask-dir",
        required=True,
        help="Directory containing *_cp_masks.tif[f] files (one per frame).",
    )
    p.add_argument(
        "--raw-dir",
        default=None,
        help="Directory with raw TIFFs for overlays (default: mask-dir).",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Name tag used in output filenames (default: v3.7_sticky_new).",
    )

    # motion / cost
    p.add_argument("--search-range", type=float, help="px per-frame drift.")
    p.add_argument("--alpha", type=float, help="distance weight.")
    p.add_argument("--beta", type=float, help="IoU weight.")
    p.add_argument("--gamma", type=float, help="area-change weight.")
    p.add_argument("--max-cost", type=float, help="max allowed assignment cost.")

    # memory
    p.add_argument("--memory", type=int, help="frames to keep track alive.")
    p.add_argument("--revive-gap", type=int, help="max gap to revive ended tracks.")

    # morphology
    p.add_argument("--dilate-radius", type=int, help="IoU dilation radius.")
    p.add_argument("--min-area", type=int, help="min object area (pixels).")

    # anti-takeover & revive
    p.add_argument(
        "--no-enforce-mutual-iou",
        action="store_true",
        help="Disable mutual IoU requirement.",
    )
    p.add_argument("--mutual-iou-tol", type=float, help="mutual IoU tolerance.")
    p.add_argument("--revive-iou-min", type=float, help="min IoU to revive.")
    p.add_argument("--no-takeover-recent", type=int, help="protect last owner N frames.")

    # hard lock
    p.add_argument("--lock-iou", type=float, help="IoU threshold for hard lock.")
    p.add_argument("--lock-dist", type=float, help="distance (px) for hard lock.")

    # rendering
    p.add_argument("--erode-px", type=int, help="not used (kept for completeness).")
    p.add_argument("--fps", type=float, help="output MP4 frame rate.")
    p.add_argument("--repeat", type=int, help="repeat each frame N times in video.")
    p.add_argument(
        "--write-track-tiffs",
        action="store_true",
        help="Write per-frame track label TIFFs.",
    )
    return p


def main():
    args = build_argparser().parse_args()

    # Use function defaults if CLI option is None.
    track_and_render(
        mask_dir=args.mask_dir,
        raw_dir=args.raw_dir,
        name=args.name or "v3.7_sticky_new",
        search_range=args.search_range if args.search_range is not None else 70.0,
        alpha=args.alpha if args.alpha is not None else 0.25,
        beta=args.beta if args.beta is not None else 0.75,
        gamma=args.gamma if args.gamma is not None else 0.05,
        max_cost=args.max_cost if args.max_cost is not None else 1.8,
        memory=args.memory if args.memory is not None else 10,
        revive_gap=args.revive_gap if args.revive_gap is not None else 60,
        dilate_radius=args.dilate_radius if args.dilate_radius is not None else 11,
        min_area=args.min_area if args.min_area is not None else 50,
        enforce_mutual_iou=not args.no_enforce_mutual_iou,
        mutual_iou_tol=args.mutual_iou_tol if args.mutual_iou_tol is not None else 1e-6,
        revive_iou_min=args.revive_iou_min if args.revive_iou_min is not None else 0.08,
        no_takeover_recent=(
            args.no_takeover_recent if args.no_takeover_recent is not None else 8
        ),
        lock_iou=args.lock_iou if args.lock_iou is not None else 0.04,
        lock_dist=args.lock_dist if args.lock_dist is not None else 60.0,
        erode_px=args.erode_px if args.erode_px is not None else 0,
        fps=args.fps if args.fps is not None else 2.0,
        repeat=args.repeat if args.repeat is not None else 3,
        write_track_tiffs=args.write_track_tiffs,
    )


if __name__ == "__main__":
    main()

