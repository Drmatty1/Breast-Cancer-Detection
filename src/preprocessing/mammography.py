"""Auditable single-image mammography preprocessing for the Stage 3 pilot.

The implementation uses only pydicom, NumPy, and Pillow. It preserves the input
raster orientation, never flips laterality, and returns review flags as exceptions
instead of silently accepting unsupported images or masks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import math

import numpy as np
import pydicom
from PIL import Image, ImageFilter


class PreprocessingError(RuntimeError):
    """Input cannot be safely decoded or transformed under this configuration."""


class DecodeError(PreprocessingError):
    """DICOM read or pixel decoding failed."""


class MalformedMetadataError(PreprocessingError):
    """Required DICOM metadata is absent or inconsistent with pixel data."""


class ReviewRequired(PreprocessingError):
    """The image needs human review before preprocessing can be accepted."""


@dataclass(frozen=True)
class PreprocessConfig:
    output_size: int = 512
    mask_max_dimension: int = 512
    otsu_bins: int = 256
    minimum_mask_coverage: float = 0.08
    maximum_mask_coverage: float = 0.92
    minimum_component_retention: float = 0.80
    minimum_bbox_extent: float = 0.10
    closing_kernel: int = 5
    expansion_kernel: int = 3
    percentile_low: float = 0.5
    percentile_high: float = 99.5
    clahe_tiles_y: int = 8
    clahe_tiles_x: int = 8
    clahe_clip_limit: float = 2.0
    crop_margin_fraction: float = 0.02
    minimum_intensity_span: float = 32.0

    def __post_init__(self) -> None:
        if self.output_size < 32 or self.mask_max_dimension < 64:
            raise ValueError("Output and mask dimensions are too small for mammograms.")
        if self.otsu_bins != 256:
            raise ValueError("This pilot implements 8-bit Otsu histograms (256 bins).")
        if not (0 <= self.percentile_low < self.percentile_high <= 100):
            raise ValueError("Invalid robust intensity percentiles.")
        if self.clahe_tiles_x < 2 or self.clahe_tiles_y < 2 or self.clahe_clip_limit <= 0:
            raise ValueError("Invalid CLAHE configuration.")
        if self.closing_kernel < 3 or self.closing_kernel % 2 == 0:
            raise ValueError("closing_kernel must be an odd integer >= 3.")
        if self.expansion_kernel < 1 or self.expansion_kernel % 2 == 0:
            raise ValueError("expansion_kernel must be odd.")


@dataclass(frozen=True)
class DecodedDicom:
    dataset: Any
    pixels: np.ndarray
    path: str


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(name, default)
        except TypeError:
            pass
    return getattr(obj, name, default)


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\\".join(str(v) for v in value)
    return str(value)


def metadata_from_dataset(ds: Any) -> dict[str, Any]:
    """Capture preprocessing-relevant metadata without applying display transforms."""
    fm = getattr(ds, "file_meta", None)
    transfer_syntax = _get(fm, "TransferSyntaxUID", "") if fm is not None else ""
    keys = (
        "Modality", "Rows", "Columns", "SamplesPerPixel", "PhotometricInterpretation",
        "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
        "PixelPaddingValue", "PixelPaddingRangeLimit", "RescaleSlope", "RescaleIntercept",
        "WindowCenter", "WindowWidth", "VOILUTSequence", "ModalityLUTSequence",
        "PresentationLUTShape", "LossyImageCompression", "PatientID", "Laterality",
        "ImageLaterality", "PatientOrientation", "ViewPosition", "ImageOrientationPatient",
        "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
    )
    result = {key: _get(ds, key) for key in keys}
    result["TransferSyntaxUID"] = str(transfer_syntax)
    return result


def decode_dataset(ds: Any) -> np.ndarray:
    """Decode one DICOM using pydicom's configured pixel decoder."""
    try:
        pixels = np.asarray(ds.pixel_array)
    except Exception as exc:
        raise DecodeError(f"DICOM pixel decoding failed: {type(exc).__name__}: {exc}") from exc
    if pixels.ndim != 2:
        raise MalformedMetadataError(f"Expected a single-channel 2D mammogram, got {pixels.shape}.")
    return pixels


def decode_dicom(path: str | Path) -> DecodedDicom:
    path = Path(path)
    try:
        ds = pydicom.dcmread(str(path), force=False)
    except Exception as exc:
        raise DecodeError(f"Could not read DICOM {path}: {type(exc).__name__}: {exc}") from exc
    return DecodedDicom(ds, decode_dataset(ds), str(path.resolve()))


def validate_pixel_metadata(pixels: np.ndarray, meta: Mapping[str, Any]) -> None:
    """Reject malformed/unsupported pixel metadata before any transformation."""
    a = np.asarray(pixels)
    if a.ndim != 2:
        raise MalformedMetadataError(f"Expected 2D grayscale pixels, got shape {a.shape}.")
    for key in ("Rows", "Columns", "PhotometricInterpretation", "BitsAllocated", "BitsStored", "PixelRepresentation", "SamplesPerPixel"):
        if meta.get(key) is None or meta.get(key) == "":
            raise MalformedMetadataError(f"Required metadata is missing: {key}.")
    if int(meta["Rows"]) != a.shape[0] or int(meta["Columns"]) != a.shape[1]:
        raise MalformedMetadataError(f"Pixel shape {a.shape} disagrees with Rows/Columns {meta['Rows']}x{meta['Columns']}.")
    if int(meta["SamplesPerPixel"]) != 1:
        raise MalformedMetadataError("Mammography pilot requires SamplesPerPixel=1.")
    if str(meta["PhotometricInterpretation"]).upper() not in {"MONOCHROME1", "MONOCHROME2"}:
        raise MalformedMetadataError(f"Unsupported photometric interpretation: {meta['PhotometricInterpretation']}.")
    bits_allocated, bits_stored, high_bit = int(meta["BitsAllocated"]), int(meta["BitsStored"]), int(meta.get("HighBit", int(meta["BitsStored"]) - 1))
    if bits_allocated not in (8, 16, 32) or not (1 <= bits_stored <= bits_allocated) or not (bits_stored - 1 <= high_bit < bits_allocated):
        raise MalformedMetadataError("BitsAllocated/BitsStored/HighBit are inconsistent.")
    if int(meta["PixelRepresentation"]) not in (0, 1):
        raise MalformedMetadataError("PixelRepresentation must be unsigned (0) or signed (1).")
    # This pilot deliberately operates on decoded stored pixels. Non-identity
    # modality transforms are not guessed or applied without dataset review.
    slope = meta.get("RescaleSlope")
    intercept = meta.get("RescaleIntercept")
    if slope not in (None, "") and not math.isclose(float(slope), 1.0):
        raise ReviewRequired(f"Non-identity RescaleSlope={slope}; modality transform requires review.")
    if intercept not in (None, "") and not math.isclose(float(intercept), 0.0):
        raise ReviewRequired(f"Non-zero RescaleIntercept={intercept}; modality transform requires review.")
    if meta.get("ModalityLUTSequence"):
        raise ReviewRequired("ModalityLUTSequence is present; stored-pixel interpretation requires review.")
    presentation = str(meta.get("PresentationLUTShape") or "").upper()
    if presentation not in ("", "IDENTITY"):
        raise ReviewRequired(f"PresentationLUTShape={presentation}; display transform requires review.")


def validate_image_provenance(ds: Any, pair: Mapping[str, Any], view: str) -> dict[str, str]:
    """Match image identity and side/view metadata to the immutable Stage 2 row."""
    role = {"CC": "cc", "MLO": "mlo"}.get(view.upper())
    if role is None:
        raise ValueError(f"View must be CC or MLO, got {view!r}.")
    patient = str(_get(ds, "PatientID", ""))
    if patient != str(pair.get("patient_id", "")):
        # CBIS DICOM PatientID is often a composite case/view token; validate
        # the manifest patient token in that composite rather than guessing.
        if str(pair.get("patient_id", "")) not in patient:
            raise ReviewRequired(f"DICOM PatientID {patient!r} does not identify manifest patient {pair.get('patient_id')!r}.")
    laterality = _string(_get(ds, "ImageLaterality") or _get(ds, "Laterality")).upper()
    expected_side = {"LEFT": "L", "RIGHT": "R"}.get(str(pair.get("breast_side", "")).upper())
    if expected_side is None or not laterality or laterality != expected_side:
        raise ReviewRequired(f"DICOM laterality {laterality!r} does not resolve manifest side {pair.get('breast_side')!r}.")
    orientation = _string(_get(ds, "PatientOrientation")).upper()
    if orientation and view.upper() not in orientation:
        raise ReviewRequired(f"DICOM PatientOrientation {orientation!r} disagrees with manifest view {view!r}.")
    uids = (str(_get(ds, "StudyInstanceUID", "")), str(_get(ds, "SeriesInstanceUID", "")))
    expected_uids = (str(pair.get(f"{role}_study_instance_uid", "")), str(pair.get(f"{role}_series_instance_uid", "")))
    if not all(expected_uids) or uids != expected_uids:
        raise ReviewRequired(f"DICOM Study/Series UIDs {uids} do not match manifest {expected_uids} for {view}.")
    return {
        "patient_id": str(pair["patient_id"]), "breast_side": str(pair["breast_side"]),
        "view": view.upper(), "study_instance_uid": uids[0], "series_instance_uid": uids[1],
        "dicom_laterality": _string(_get(ds, "Laterality")),
        "dicom_image_laterality": _string(_get(ds, "ImageLaterality")),
        "orientation_status": "source_view_and_dicom_laterality_match; original raster preserved without flipping",
    }


def validate_pair_record(row: Mapping[str, Any]) -> None:
    """Check pair completeness and train-only status before pilot processing."""
    required = ("pair_id", "patient_id", "breast_side", "partition", "cc_local_path", "mlo_local_path",
                "cc_study_instance_uid", "cc_series_instance_uid", "mlo_study_instance_uid", "mlo_series_instance_uid")
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise MalformedMetadataError(f"Incomplete pair row; missing {missing}.")
    if row["partition"] != "train":
        raise ReviewRequired(f"Pilot inputs must be training partition only, got {row['partition']} for {row['patient_id']}.")
    if row["breast_side"] not in ("LEFT", "RIGHT"):
        raise MalformedMetadataError(f"Invalid breast side {row['breast_side']!r}.")
    if (row["cc_study_instance_uid"], row["cc_series_instance_uid"]) == (row["mlo_study_instance_uid"], row["mlo_series_instance_uid"]):
        raise MalformedMetadataError("CC and MLO source UID pairs must differ.")
    if not Path(row["cc_local_path"]).is_file() or not Path(row["mlo_local_path"]).is_file():
        raise FileNotFoundError(f"A local full-mammogram file is missing for {row['pair_id']}.")


def valid_pixel_mask(pixels: np.ndarray, padding_value: Any = None, padding_range_limit: Any = None) -> np.ndarray:
    a = np.asarray(pixels)
    valid = np.isfinite(a)
    if padding_value is not None:
        pad = float(padding_value)
        if padding_range_limit is None:
            valid &= a != pad
        else:
            limit = float(padding_range_limit)
            valid &= ~((a >= min(pad, limit)) & (a <= max(pad, limit)))
    return valid


def normalize_polarity(pixels: np.ndarray, photometric: str, valid: np.ndarray) -> np.ndarray:
    """Map MONOCHROME1 to MONOCHROME2 display polarity over valid stored values."""
    a = np.asarray(pixels, dtype=np.float32)
    photo = str(photometric).upper()
    if photo == "MONOCHROME2":
        return np.where(valid, a, 0.0).astype(np.float32)
    if photo != "MONOCHROME1":
        raise MalformedMetadataError(f"Unsupported photometric interpretation {photometric!r}.")
    values = a[valid]
    if values.size == 0:
        raise ReviewRequired("No valid pixel values remain after padding/non-finite exclusion.")
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        raise ReviewRequired("MONOCHROME1 image has no usable intensity range.")
    out = np.zeros_like(a, dtype=np.float32)
    out[valid] = lo + hi - a[valid]
    return out


def otsu_threshold_u8(image: np.ndarray, valid: np.ndarray | None = None) -> int:
    a = np.asarray(image, dtype=np.uint8)
    if a.ndim != 2:
        raise ValueError("Otsu input must be a 2D uint8 image.")
    values = a.ravel() if valid is None else a[np.asarray(valid, dtype=bool)]
    if values.size == 0:
        raise ReviewRequired("No valid coarse pixels available for Otsu thresholding.")
    hist = np.bincount(values, minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256, dtype=np.float64))
    total_mu = mu[-1]
    denom = omega * (1.0 - omega)
    between = np.zeros(256, dtype=np.float64)
    good = denom > 0
    between[good] = (total_mu * omega[good] - mu[good]) ** 2 / denom[good]
    return int(np.argmax(between))


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return the 8-connected largest component using a deterministic flood fill."""
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2 or not source.any():
        raise ReviewRequired("Breast mask is empty or malformed.")
    h, w = source.shape
    seen = np.zeros(source.shape, dtype=bool)
    flat = source.ravel()
    seeds = np.flatnonzero(flat)
    best: list[int] = []
    for seed_np in seeds:
        seed = int(seed_np)
        if seen.ravel()[seed]:
            continue
        queue = [seed]
        seen.ravel()[seed] = True
        component: list[int] = []
        head = 0
        while head < len(queue):
            pos = queue[head]; head += 1
            component.append(pos)
            y, x = divmod(pos, w)
            y0, y1 = max(0, y - 1), min(h - 1, y + 1)
            x0, x1 = max(0, x - 1), min(w - 1, x + 1)
            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    q = base + xx
                    if flat[q] and not seen.ravel()[q]:
                        seen.ravel()[q] = True
                        queue.append(q)
        if len(component) > len(best):
            best = component
    out = np.zeros(source.size, dtype=bool)
    out[np.asarray(best, dtype=np.int64)] = True
    return out.reshape(source.shape), len(best), int(source.sum())


def _morph_cleanup(mask: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    closed = image.filter(ImageFilter.MaxFilter(config.closing_kernel)).filter(ImageFilter.MinFilter(config.closing_kernel))
    expanded = closed.filter(ImageFilter.MaxFilter(config.expansion_kernel))
    return np.asarray(expanded, dtype=np.uint8) > 0


def validate_breast_mask(mask: np.ndarray, raw_foreground_count: int, component_count: int, config: PreprocessConfig) -> float:
    m = np.asarray(mask, dtype=bool)
    coverage = float(m.mean()) if m.size else 0.0
    retention = float(component_count / raw_foreground_count) if raw_foreground_count else 0.0
    if not m.any():
        raise ReviewRequired("Detected breast mask is empty.")
    if not (config.minimum_mask_coverage <= coverage <= config.maximum_mask_coverage):
        raise ReviewRequired(f"Implausible breast-mask coverage {coverage:.4f}.")
    if retention < config.minimum_component_retention:
        raise ReviewRequired(f"Largest component retains only {retention:.3f} of threshold foreground.")
    ys, xs = np.nonzero(m)
    if ((ys.max() - ys.min() + 1) / m.shape[0] < config.minimum_bbox_extent or
            (xs.max() - xs.min() + 1) / m.shape[1] < config.minimum_bbox_extent):
        raise ReviewRequired("Breast-mask bounding box is implausibly small.")
    return retention


def clahe_u8(image: np.ndarray, mask: np.ndarray, tiles_y: int = 8, tiles_x: int = 8, clip_limit: float = 2.0) -> np.ndarray:
    """Deterministic mask-aware CLAHE with bilinear interpolation between tile LUTs."""
    src = np.asarray(image, dtype=np.uint8)
    valid = np.asarray(mask, dtype=bool)
    if src.ndim != 2 or valid.shape != src.shape:
        raise ValueError("CLAHE expects matching 2D image and mask arrays.")
    if not valid.any():
        raise ReviewRequired("CLAHE received an empty breast mask.")
    h, w = src.shape
    ny, nx = min(tiles_y, h), min(tiles_x, w)
    y_edges = np.linspace(0, h, ny + 1, dtype=int)
    x_edges = np.linspace(0, w, nx + 1, dtype=int)
    y_centers = (y_edges[:-1] + y_edges[1:] - 1) / 2.0
    x_centers = (x_edges[:-1] + x_edges[1:] - 1) / 2.0
    luts = np.empty((ny, nx, 256), dtype=np.uint8)
    for ty in range(ny):
        for tx in range(nx):
            y0, y1 = y_edges[ty], y_edges[ty + 1]
            x0, x1 = x_edges[tx], x_edges[tx + 1]
            use = valid[y0:y1, x0:x1]
            vals = src[y0:y1, x0:x1][use]
            if vals.size == 0:
                luts[ty, tx] = np.arange(256, dtype=np.uint8)
                continue
            hist = np.bincount(vals, minlength=256).astype(np.int64)
            limit = max(1, int(clip_limit * vals.size / 256.0))
            excess = int(np.maximum(hist - limit, 0).sum())
            hist = np.minimum(hist, limit)
            hist += excess // 256
            remainder = excess % 256
            if remainder:
                hist[np.linspace(0, 255, remainder, endpoint=False, dtype=int)] += 1
            cdf = np.cumsum(hist, dtype=np.float64)
            if cdf[-1] <= 0:
                luts[ty, tx] = np.arange(256, dtype=np.uint8)
            else:
                luts[ty, tx] = np.clip(np.rint(cdf * (255.0 / cdf[-1])), 0, 255).astype(np.uint8)

    out = np.zeros_like(src)
    y_index = np.arange(h)
    x_index = np.arange(w)
    for ty in range(ny):
        y0, y1 = y_edges[ty], y_edges[ty + 1]
        y = y_index[y0:y1]
        yi1 = np.searchsorted(y_centers, y, side="right")
        yi0 = np.clip(yi1 - 1, 0, ny - 1)
        yi1 = np.clip(yi1, 0, ny - 1)
        wy = np.zeros_like(y, dtype=np.float32)
        distinct = yi0 != yi1
        wy[distinct] = ((y[distinct] - y_centers[yi0[distinct]]) /
                        (y_centers[yi1[distinct]] - y_centers[yi0[distinct]])).astype(np.float32)
        for tx in range(nx):
            x0, x1 = x_edges[tx], x_edges[tx + 1]
            x = x_index[x0:x1]
            xi1 = np.searchsorted(x_centers, x, side="right")
            xi0 = np.clip(xi1 - 1, 0, nx - 1)
            xi1 = np.clip(xi1, 0, nx - 1)
            wx = np.zeros_like(x, dtype=np.float32)
            distinct_x = xi0 != xi1
            wx[distinct_x] = ((x[distinct_x] - x_centers[xi0[distinct_x]]) /
                              (x_centers[xi1[distinct_x]] - x_centers[xi0[distinct_x]])).astype(np.float32)
            for local_y, global_y in enumerate(y):
                vals = src[global_y, x0:x1]
                iy0, iy1, fy = int(yi0[local_y]), int(yi1[local_y]), float(wy[local_y])
                a = luts[iy0, xi0, vals]
                b = luts[iy0, xi1, vals]
                c = luts[iy1, xi0, vals]
                d = luts[iy1, xi1, vals]
                mapped = (a * (1-fy) * (1-wx) + b * (1-fy) * wx +
                          c * fy * (1-wx) + d * fy * wx)
                row = out[global_y, x0:x1]
                row[:] = np.clip(np.rint(mapped), 0, 255).astype(np.uint8)
                row[~valid[global_y, x0:x1]] = 0
    return out


def preprocess_pixels(pixels: np.ndarray, metadata: Mapping[str, Any], config: PreprocessConfig | None = None) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Preprocess one stored-pixel mammogram and return image, metrics, and QA stages."""
    config = config or PreprocessConfig()
    a = np.asarray(pixels)
    validate_pixel_metadata(a, metadata)
    valid = valid_pixel_mask(a, metadata.get("PixelPaddingValue"), metadata.get("PixelPaddingRangeLimit"))
    if not valid.any():
        raise ReviewRequired("No valid stored pixel values remain.")
    raw_valid = a[valid].astype(np.float32)
    raw_stats = {
        "raw_valid_min": float(np.min(raw_valid)), "raw_valid_max": float(np.max(raw_valid)),
        "raw_valid_mean": float(np.mean(raw_valid)), "raw_valid_std": float(np.std(raw_valid)),
        "raw_valid_p01": float(np.percentile(raw_valid, 1)), "raw_valid_p50": float(np.percentile(raw_valid, 50)),
        "raw_valid_p99": float(np.percentile(raw_valid, 99)),
    }
    displayed = normalize_polarity(a, str(metadata["PhotometricInterpretation"]), valid)
    pre_low, pre_high = np.percentile(displayed[valid], [config.percentile_low, config.percentile_high])
    if not np.isfinite(pre_high - pre_low) or pre_high - pre_low < config.minimum_intensity_span:
        raise ReviewRequired(f"Constant or near-constant mammogram intensity span {pre_high-pre_low:.3f}.")
    provisional = np.zeros(a.shape, dtype=np.float32)
    provisional[valid] = np.clip((displayed[valid] - pre_low) / (pre_high - pre_low), 0.0, 1.0)
    h, w = a.shape
    scale = min(1.0, config.mask_max_dimension / max(h, w))
    coarse_w = max(1, int(round(w * scale)))
    coarse_h = max(1, int(round(h * scale)))
    coarse_gray = np.asarray(Image.fromarray(np.rint(provisional * 255).astype(np.uint8), mode="L").resize((coarse_w, coarse_h), Image.Resampling.BILINEAR))
    coarse_valid = np.asarray(Image.fromarray(valid.astype(np.uint8) * 255, mode="L").resize((coarse_w, coarse_h), Image.Resampling.NEAREST)) > 0
    threshold = otsu_threshold_u8(coarse_gray, coarse_valid)
    threshold_mask = (coarse_gray > threshold) & coarse_valid
    cleaned = _morph_cleanup(threshold_mask, config)
    coarse_mask, component_count, _ = largest_component(cleaned)
    retained_threshold_count = int((coarse_mask & threshold_mask).sum())
    retention = validate_breast_mask(coarse_mask, int(threshold_mask.sum()), retained_threshold_count, config)
    mask_full = np.asarray(Image.fromarray(coarse_mask.astype(np.uint8) * 255, mode="L").resize((w, h), Image.Resampling.NEAREST)) > 0
    mask_full &= valid
    if not mask_full.any():
        raise ReviewRequired("Coarse breast mask became empty when mapped to source dimensions.")
    mask_valid_values = displayed[mask_full]
    low, high = np.percentile(mask_valid_values, [config.percentile_low, config.percentile_high])
    if high - low < config.minimum_intensity_span:
        raise ReviewRequired(f"Breast-region intensity span {high-low:.3f} is too small.")
    normalized = np.zeros(a.shape, dtype=np.float32)
    normalized[mask_full] = np.clip((displayed[mask_full] - low) / (high - low), 0.0, 1.0)
    clip_low = float(np.mean(displayed[mask_full] < low))
    clip_high = float(np.mean(displayed[mask_full] > high))
    normalized_u8 = np.rint(normalized * 255).astype(np.uint8)
    enhanced_full = clahe_u8(normalized_u8, mask_full, config.clahe_tiles_y, config.clahe_tiles_x, config.clahe_clip_limit)
    ys, xs = np.nonzero(mask_full)
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    margin = max(2, int(round(max(x1-x0, y1-y0) * config.crop_margin_fraction)))
    x0, x1 = max(0, x0-margin), min(w, x1+margin)
    y0, y1 = max(0, y0-margin), min(h, y1+margin)
    crop = enhanced_full[y0:y1, x0:x1]
    crop_mask = mask_full[y0:y1, x0:x1]
    ch, cw = crop.shape
    factor = min(config.output_size / cw, config.output_size / ch)
    rw, rh = max(1, int(round(cw * factor))), max(1, int(round(ch * factor)))
    resampled = Image.fromarray(crop, mode="L").resize((rw, rh), Image.Resampling.LANCZOS)
    resized_mask = Image.fromarray(crop_mask.astype(np.uint8) * 255, mode="L").resize((rw, rh), Image.Resampling.NEAREST)
    canvas = Image.new("L", (config.output_size, config.output_size), 0)
    mask_canvas = Image.new("L", (config.output_size, config.output_size), 0)
    left, top = (config.output_size-rw)//2, (config.output_size-rh)//2
    canvas.paste(resampled, (left, top)); mask_canvas.paste(resized_mask, (left, top))
    out = np.asarray(canvas, dtype=np.float32) / 255.0
    out_mask = np.asarray(mask_canvas, dtype=np.uint8) > 0
    out[~out_mask] = 0.0
    if out.shape != (config.output_size, config.output_size) or out.dtype != np.float32 or not np.isfinite(out).all():
        raise PreprocessingError("Output shape/dtype/finite-value validation failed.")
    bbox_fraction = float(((xs.max()+1-xs.min()) * (ys.max()+1-ys.min())) / (w*h))
    metrics: dict[str, Any] = {
        **raw_stats,
        "original_rows": int(h), "original_columns": int(w), "pixel_dtype": str(a.dtype),
        "photometric_interpretation": str(metadata["PhotometricInterpretation"]),
        "bits_allocated": int(metadata["BitsAllocated"]), "bits_stored": int(metadata["BitsStored"]),
        "high_bit": int(metadata.get("HighBit", int(metadata["BitsStored"])-1)),
        "pixel_representation": int(metadata["PixelRepresentation"]),
        "padding_value": metadata.get("PixelPaddingValue"),
        "padding_range_limit": metadata.get("PixelPaddingRangeLimit"),
        "invalid_pixel_fraction": float(1.0-valid.mean()),
        "threshold_u8": int(threshold), "threshold_foreground_coverage_coarse": float(threshold_mask.mean()),
        "largest_component_coverage_coarse": float(coarse_mask.mean()), "component_retention": retention,
        "breast_bbox_fraction_source": bbox_fraction,
        "mask_coverage_source": float(mask_full.mean()), "mask_coverage_output": float(out_mask.mean()),
        "normalization_percentiles": [float(config.percentile_low), float(config.percentile_high)],
        "normalization_low_stored_value": float(low), "normalization_high_stored_value": float(high),
        "clipped_low_fraction_in_mask": clip_low, "clipped_high_fraction_in_mask": clip_high,
        "clahe_tiles": [int(config.clahe_tiles_y), int(config.clahe_tiles_x)],
        "clahe_clip_limit": float(config.clahe_clip_limit),
        "crop_xyxy_source": [x0, y0, x1, y1], "crop_rows": int(ch), "crop_columns": int(cw),
        "resized_rows": int(rh), "resized_columns": int(rw),
        "padding_top": int(top), "padding_bottom": int(config.output_size-rh-top),
        "padding_left": int(left), "padding_right": int(config.output_size-rw-left),
        "aspect_ratio_preserved": True, "horizontal_flip": False,
        "output_shape": list(out.shape), "output_dtype": str(out.dtype),
        "output_min": float(out.min()), "output_max": float(out.max()),
        "dataset_mean_std_normalization": "deferred_until_full_training_cohort_statistics_are_approved",
    }
    qa = {
        "normalized_source": provisional.astype(np.float32),
        "mask_source": mask_full.astype(bool),
        "masked_source": (normalized * mask_full).astype(np.float32),
        "clahe_source": enhanced_full,
        "output_mask": out_mask,
    }
    return out, metrics, qa


def preprocess_dicom(path: str | Path, pair: Mapping[str, Any], view: str, config: PreprocessConfig | None = None) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    validate_pair_record(pair)
    role = {"CC": "cc", "MLO": "mlo"}[view.upper()]
    path = Path(pair[f"{role}_local_path"])
    decoded = decode_dicom(path)
    provenance = validate_image_provenance(decoded.dataset, pair, view)
    metadata = metadata_from_dataset(decoded.dataset)
    validate_pixel_metadata(decoded.pixels, metadata)
    image, metrics, qa = preprocess_pixels(decoded.pixels, metadata, config)
    metrics.update(provenance)
    metrics.update({
        "transfer_syntax_uid": metadata["TransferSyntaxUID"],
        "modality": _string(metadata["Modality"]),
        "rescale_slope_recorded_not_applied": _string(metadata["RescaleSlope"]),
        "rescale_intercept_recorded_not_applied": _string(metadata["RescaleIntercept"]),
        "window_center_recorded_not_applied": _string(metadata["WindowCenter"]),
        "window_width_recorded_not_applied": _string(metadata["WindowWidth"]),
        "voi_lut_present_not_applied": bool(metadata["VOILUTSequence"]),
        "modality_lut_present": bool(metadata["ModalityLUTSequence"]),
        "presentation_lut_shape_recorded_not_applied": _string(metadata["PresentationLUTShape"]),
        "patient_orientation": _string(metadata["PatientOrientation"]),
        "view_position": _string(metadata["ViewPosition"]),
        "image_orientation_patient": _string(metadata["ImageOrientationPatient"]),
        "local_dicom_path": str(path.resolve()),
        "source_path": pair.get(f"{role}_source_path", ""),
        "source_study_instance_uid": pair[f"{role}_study_instance_uid"],
        "source_series_instance_uid": pair[f"{role}_series_instance_uid"],
        "sop_instance_uid": _string(metadata["SOPInstanceUID"]),
        "pair_id": pair["pair_id"], "partition": pair["partition"],
    })
    return image, metrics, qa


def config_as_dict(config: PreprocessConfig) -> dict[str, Any]:
    return asdict(config)
