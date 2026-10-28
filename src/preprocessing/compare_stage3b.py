"""Run a deterministic three-configuration comparison on the Stage 3 train pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing.mammography import PreprocessConfig, preprocess_dicom, validate_pair_record  # noqa: E402
from preprocessing.pilot import read_csv, write_csv  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_pairs(root: Path) -> list[dict[str, str]]:
    rows = read_csv(root / "data/manifests/stage2/cohort_pairs.csv")
    splits = {r["patient_id"]: r["partition"] for r in read_csv(root / "data/manifests/stage2/patient_splits.csv")}
    ids = ("P_00008", "P_00021", "P_00112")
    selected = [r for r in rows if r["patient_id"] in ids]
    if len(selected) != 6 or {r["patient_id"] for r in selected} != set(ids):
        raise RuntimeError(f"Expected exactly six pairs for {ids}; found {len(selected)}.")
    for r in selected:
        if splits.get(r["patient_id"]) != "train" or r["partition"] != "train":
            raise RuntimeError(f"Non-training pilot input detected: {r['patient_id']} / {r['partition']}.")
        validate_pair_record(r)
    return sorted(selected, key=lambda r: (r["patient_id"], r["breast_side"]))


def quant_metrics(image: np.ndarray, mask: np.ndarray) -> dict[str, float | int | str]:
    a = np.asarray(image)
    m = np.asarray(mask, dtype=bool)
    vals = a[m]
    if not vals.size:
        raise RuntimeError("Empty output mask in comparison.")
    # A simple high-frequency proxy: standard deviation of horizontal/vertical
    # first differences whose endpoints are both inside the output breast mask.
    dx = np.diff(a, axis=1)
    mx = m[:, 1:] & m[:, :-1]
    dy = np.diff(a, axis=0)
    my = m[1:, :] & m[:-1, :]
    grads = np.concatenate([dx[mx], dy[my]]) if mx.any() or my.any() else np.array([], dtype=np.float32)
    yy, xx = np.nonzero(m)
    return {
        "output_shape": "x".join(map(str, a.shape)), "dtype": str(a.dtype),
        "finite_fraction": float(np.isfinite(a).mean()), "mask_coverage": float(m.mean()),
        "background_nonzero_fraction": float(np.count_nonzero(a[~m]) / max(1, (~m).sum())),
        "breast_p01": float(np.percentile(vals, 1)), "breast_p50": float(np.percentile(vals, 50)),
        "breast_p99": float(np.percentile(vals, 99)), "breast_std": float(vals.std()),
        "high_frequency_gradient_std": float(grads.std()) if grads.size else 0.0,
        "mask_edge_touch_fraction": float((m[0].sum()+m[-1].sum()+m[:,0].sum()+m[:,-1].sum()) / max(1, 2*m.shape[0]+2*m.shape[1])),
        "mask_bbox_aspect_ratio": float((xx.max()-xx.min()+1)/(yy.max()-yy.min()+1)),
    }


def comparison_image(outputs: dict[str, np.ndarray], masks: dict[str, np.ndarray], path: Path, title: str) -> None:
    labels = ["A baseline", "B clip=1.0", "C tight breast crop"]
    board = Image.new("RGB", (3*512, 2*512+64), (24, 24, 24))
    draw = ImageDraw.Draw(board)
    for i, key in enumerate(("A", "B", "C")):
        arr = np.asarray(outputs[key], dtype=np.float32)
        preview = np.clip(np.rint(arr*255), 0, 255).astype(np.uint8)
        panel = Image.fromarray(preview, mode="L").convert("RGB")
        board.paste(panel, (i*512, 32))
        draw.text((i*512+8, 7), f"{labels[i]} | mask={masks[key].mean():.1%}", fill="white", font=ImageFont.load_default())
        m = Image.fromarray(np.asarray(masks[key], dtype=np.uint8)*255, mode="L").convert("RGB")
        board.paste(m, (i*512, 544))
        draw.text((i*512+8, 521), f"{labels[i]} detected output mask", fill="white", font=ImageFont.load_default())
    draw.text((8, 1058), f"{title} | same source raster; no flips", fill="white", font=ImageFont.load_default())
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path, format="PNG", optimize=True)


def run(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    out_root = root / "data/processed/stage3b_comparison"
    qa_root = root / "reports/stage3b/visual_qa"
    report_root = root / "reports/stage3b"
    for path in (out_root, qa_root, report_root):
        path.mkdir(parents=True, exist_ok=True)
    if any(out_root.iterdir()) or any(qa_root.iterdir()):
        raise FileExistsError("Stage 3B output folder is not empty; refusing to overwrite.")

    configs = {
        "A": PreprocessConfig(),
        "B": PreprocessConfig(clahe_clip_limit=1.0),
        "C": PreprocessConfig(crop_margin_fraction=0.005),
    }
    cohort = root / "data/manifests/stage2/cohort_pairs.csv"
    splits = root / "data/manifests/stage2/patient_splits.csv"
    manifest_hashes = {str(p): sha256(p) for p in (cohort, splits)}
    pairs = select_pairs(root)
    sources = sorted({Path(r[k]) for r in pairs for k in ("cc_local_path", "mlo_local_path")})
    source_hashes = {str(p): sha256(p) for p in sources}
    outputs: dict[str, dict[str, np.ndarray]] = {k: {} for k in configs}
    masks: dict[str, dict[str, np.ndarray]] = {k: {} for k in configs}
    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    qa_paths: list[Path] = []
    failures: list[dict[str, str]] = []
    for pair in pairs:
        per_pair = {k: {} for k in configs}
        pair_id = pair["pair_id"]
        try:
            for view, role in (("CC", "cc"), ("MLO", "mlo")):
                source = Path(pair[f"{role}_local_path"])
                view_outputs = {}
                view_masks = {}
                view_metrics = {}
                for name, cfg in configs.items():
                    arr, metrics, qa = preprocess_dicom(source, pair, view, cfg)
                    view_outputs[name] = arr
                    view_masks[name] = qa["output_mask"]
                    view_metrics[name] = metrics
                # Same segmentation/normalization should hold across CLAHE and crop variants.
                if len({view_metrics[k]["mask_coverage_source"] for k in configs}) != 1:
                    raise RuntimeError("Unexpected source mask difference across configurations.")
                key = f"{pair_id}_{view}"
                for name in configs:
                    outputs[name][key] = view_outputs[name]
                    masks[name][key] = view_masks[name]
                    target = out_root / name / pair["patient_id"] / pair["breast_side"] / f"{view}.npy"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.save(target, view_outputs[name], allow_pickle=False)
                    base = {
                        "configuration": name, "pair_id": pair_id, "patient_id": pair["patient_id"],
                        "partition": pair["partition"], "breast_side": pair["breast_side"], "view": view,
                        "source_dicom_path": str(source.resolve()), "study_instance_uid": view_metrics[name]["study_instance_uid"],
                        "series_instance_uid": view_metrics[name]["series_instance_uid"], "sop_instance_uid": view_metrics[name]["sop_instance_uid"],
                        "output_path": str(target.resolve()), "output_sha256": sha256(target),
                        "config_json": json.dumps({**cfg_dict(configs[name]), "variant": name}, sort_keys=True),
                    }
                    metric_rows.append({**base, **view_metrics[name], **quant_metrics(view_outputs[name], view_masks[name])})
                png = qa_root / f"{pair_id}_{view}_ABC.png"
                comparison_image(view_outputs, view_masks, png, f"{pair_id} / {view}")
                qa_paths.append(png)
            pair_rows.append({"pair_id": pair_id, "patient_id": pair["patient_id"], "breast_side": pair["breast_side"], "partition": pair["partition"], "status": "complete"})
        except Exception as exc:
            failures.append({"pair_id": pair_id, "error_type": type(exc).__name__, "error": str(exc)})
            pair_rows.append({"pair_id": pair_id, "patient_id": pair["patient_id"], "breast_side": pair["breast_side"], "partition": pair["partition"], "status": "failed", "error": str(exc)})
    post_hashes = {str(p): sha256(p) for p in (cohort, splits)}
    post_sources = {str(p): sha256(p) for p in sources}
    if post_hashes != manifest_hashes or post_sources != source_hashes:
        raise RuntimeError("Source files or Stage 2 manifests changed during comparison.")
    write_csv(out_root / "comparison_metrics.csv", metric_rows, list(metric_rows[0]) if metric_rows else ["configuration"])
    write_csv(out_root / "pair_status.csv", pair_rows, list(pair_rows[0]) if pair_rows else ["pair_id"])
    summary = {
        "patients": ["P_00008", "P_00021", "P_00112"], "partition": "train", "pair_count": 6,
        "view_count_per_configuration": len(metric_rows)//3, "configurations": {k: cfg_dict(v) for k,v in configs.items()},
        "failures": failures, "stage2_manifest_hashes_unchanged": manifest_hashes == post_hashes,
        "source_dicom_hashes_unchanged": source_hashes == post_sources,
        "source_image_count": len(sources), "visual_comparison_count": len(qa_paths),
    }
    (out_root / "comparison_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return summary


def cfg_dict(cfg: PreprocessConfig) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(cfg)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
