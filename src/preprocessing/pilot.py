"""Run a deterministic, training-only Stage 3 preprocessing pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing.mammography import (  # noqa: E402
    PreprocessConfig, PreprocessingError, config_as_dict, preprocess_dicom,
    validate_pair_record,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_panel(array: np.ndarray, panel_size: int = 512, is_mask: bool = False) -> Image.Image:
    if is_mask:
        src = np.asarray(array, dtype=np.uint8) * 255
    else:
        src = np.clip(np.rint(np.asarray(array, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(src, mode="L")
    h, w = src.shape
    scale = min(panel_size / w, panel_size / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    resized = image.resize(size, resampling)
    canvas = Image.new("L", (panel_size, panel_size), 0)
    canvas.paste(resized, ((panel_size-size[0])//2, (panel_size-size[1])//2))
    return canvas


def save_four_panel(pair_id: str, view: str, qa: dict[str, np.ndarray], output: np.ndarray, path: Path) -> None:
    size, header = 512, 34
    board = Image.new("RGB", (size*4, size+header), (20, 20, 20))
    draw = ImageDraw.Draw(board)
    panels = (
        ("Original raster", fit_panel(qa["normalized_source"], size)),
        ("Detected breast mask", fit_panel(qa["mask_source"], size, True)),
        ("Masked source", fit_panel(qa["masked_source"], size)),
        ("Final CLAHE 512x512", fit_panel(output, size)),
    )
    for i, (title, panel) in enumerate(panels):
        draw.text((i*size+10, 8), title, fill=(245, 245, 245), font=ImageFont.load_default())
        board.paste(panel.convert("RGB"), (i*size, header))
    draw.text((8, size+header-15), f"{pair_id} / {view} / native raster preserved; no flip", fill=(230, 230, 230))
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path, format="PNG", optimize=True)


def save_contact_sheet(paths: list[Path], path: Path) -> None:
    thumbs=[]
    for src in paths:
        im=Image.open(src).convert("RGB")
        im.thumbnail((1024, 300), Image.Resampling.LANCZOS)
        thumbs.append((src.name, im.copy()))
        im.close()
    cols=2; rows=(len(thumbs)+cols-1)//cols; cell_w=1024; cell_h=324
    sheet=Image.new("RGB", (cols*cell_w, rows*cell_h), (30,30,30)); draw=ImageDraw.Draw(sheet)
    for i,(name,im) in enumerate(thumbs):
        x=(i%cols)*cell_w; y=(i//cols)*cell_h
        draw.text((x+6,y+4),name,fill=(255,255,255),font=ImageFont.load_default())
        sheet.paste(im,(x,y+22))
    sheet.save(path,format="PNG",optimize=True)


def deterministic_sample(pair_rows: list[dict[str, str]], split_rows: list[dict[str, str]], count: int=3) -> tuple[list[str], list[dict[str,str]]]:
    split={r["patient_id"]:r["partition"] for r in split_rows}
    sides=defaultdict(set)
    for row in pair_rows:
        if row["patient_id"] not in split:
            raise PreprocessingError(f"Pair patient {row['patient_id']} is absent from patient_splits.csv.")
        if row["partition"] != split[row["patient_id"]]:
            raise PreprocessingError(f"Pair/split partition mismatch for {row['patient_id']}.")
        sides[row["patient_id"]].add(row["breast_side"])
    selected=[pid for pid in sorted(sides) if sides[pid]=={"LEFT","RIGHT"} and split[pid]=="train"][:count]
    if len(selected)<count:
        raise PreprocessingError(f"Only {len(selected)} bilateral training patients are available; expected {count}.")
    result=[r for r in pair_rows if r["patient_id"] in selected]
    if len(result)!=count*2:
        raise PreprocessingError(f"Bilateral pilot sample is incomplete: expected {count*2} pair rows, got {len(result)}.")
    return selected, sorted(result,key=lambda r:(r["patient_id"],r["breast_side"]))


def run_pilot(root: Path, config_path: Path) -> dict[str, Any]:
    root=root.resolve()
    cohort_path=root/"data/manifests/stage2/cohort_pairs.csv"
    splits_path=root/"data/manifests/stage2/patient_splits.csv"
    out_root=root/"data/processed/stage3_pilot"
    qa_root=root/"reports/stage3/visual_qa"
    for path in (cohort_path,splits_path,config_path):
        if not path.is_file(): raise FileNotFoundError(path)
    if any(out_root.iterdir()) or any(qa_root.iterdir()):
        raise FileExistsError("Stage 3 pilot output folders are not empty; refusing to overwrite.")
    raw_cfg=json.loads(config_path.read_text(encoding="utf-8-sig"))
    allowed={k:v for k,v in raw_cfg.items() if k in PreprocessConfig.__dataclass_fields__}
    config=PreprocessConfig(**allowed)
    pair_rows=read_csv(cohort_path); split_rows=read_csv(splits_path)
    selected, sample=deterministic_sample(pair_rows,split_rows,3)
    for row in sample: validate_pair_record(row)
    original_hashes={str(p):file_sha256(p) for p in (cohort_path,splits_path)}
    source_paths={Path(r[k]) for r in sample for k in ("cc_local_path","mlo_local_path")}
    source_stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in source_paths}
    source_hashes={str(p):file_sha256(p) for p in source_paths}
    image_rows=[]; pair_manifest=[]; qa_paths=[]; failures=[]
    for pair in sample:
        pair_outputs={}; pair_metrics={}; pair_qas={}; pair_failures=[]
        for view,role in (("CC","cc"),("MLO","mlo")):
            try:
                arr,metrics,qa=preprocess_dicom(pair[f"{role}_local_path"],pair,view,config)
                pair_outputs[view]=arr; pair_metrics[view]=metrics; pair_qas[view]=qa
            except Exception as exc:
                pair_failures.append({"view":view,"error_type":type(exc).__name__,"error":str(exc)})
        if pair_failures:
            failures.append({"pair_id":pair["pair_id"],"patient_id":pair["patient_id"],"breast_side":pair["breast_side"],"failures":pair_failures})
            pair_manifest.append({"pair_id":pair["pair_id"],"patient_id":pair["patient_id"],"breast_side":pair["breast_side"],
              "partition":pair["partition"],"status":"review_required_or_failed","cc_uid_pair":f"{pair['cc_study_instance_uid']}|{pair['cc_series_instance_uid']}",
              "mlo_uid_pair":f"{pair['mlo_study_instance_uid']}|{pair['mlo_series_instance_uid']}",
              "cc_output_path":"","mlo_output_path":"","failure_json":json.dumps(pair_failures,ensure_ascii=False)})
            continue
        saved_paths={}
        # Write neither view until both independently pass image and provenance checks.
        for view in ("CC","MLO"):
            metrics=pair_metrics[view]; arr=pair_outputs[view]
            output_path=out_root/pair["patient_id"]/pair["breast_side"]/f"{view}.npy"
            if output_path.exists(): raise FileExistsError(output_path)
            output_path.parent.mkdir(parents=True,exist_ok=True)
            np.save(output_path,arr,allow_pickle=False)
            qa_path=qa_root/f"{pair['pair_id']}_{view}.png"
            save_four_panel(pair["pair_id"],view,pair_qas[view],arr,qa_path)
            qa_paths.append(qa_path); saved_paths[view]=str(output_path.resolve())
            image_rows.append({
              "pair_id":pair["pair_id"],"patient_id":pair["patient_id"],"breast_side":pair["breast_side"],
              "partition":pair["partition"],"view":view,"source_dicom_path":metrics["local_dicom_path"],
              "source_csv_path":metrics["source_path"],"study_instance_uid":metrics["study_instance_uid"],
              "series_instance_uid":metrics["series_instance_uid"],"sop_instance_uid":metrics["sop_instance_uid"],
              "output_path":str(output_path.resolve()),"visual_qa_path":str(qa_path.resolve()),
              "status":"processed","metrics_json":json.dumps(metrics,ensure_ascii=False,sort_keys=True),
            })
        pair_manifest.append({"pair_id":pair["pair_id"],"patient_id":pair["patient_id"],"breast_side":pair["breast_side"],
          "partition":pair["partition"],"status":"processed","cc_uid_pair":f"{pair['cc_study_instance_uid']}|{pair['cc_series_instance_uid']}",
          "mlo_uid_pair":f"{pair['mlo_study_instance_uid']}|{pair['mlo_series_instance_uid']}",
          "cc_output_path":saved_paths["CC"],"mlo_output_path":saved_paths["MLO"],"failure_json":""})
    contact_path=qa_root/"pilot_visual_qa_contact_sheet.png"
    if qa_paths: save_contact_sheet(qa_paths,contact_path)
    after_hashes={str(p):file_sha256(p) for p in (cohort_path,splits_path)}
    after_source_stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in source_paths}
    after_source_hashes={str(p):file_sha256(p) for p in source_paths}
    if original_hashes!=after_hashes or source_stats!=after_source_stats or source_hashes!=after_source_hashes:
        raise PreprocessingError("A Stage 2 manifest or pilot source DICOM changed during the run.")
    if image_rows:
        fields=list(image_rows[0]); write_csv(out_root/"pilot_manifest.csv",image_rows,fields)
    else:
        fields=["pair_id","patient_id","breast_side","partition","view","source_dicom_path","source_csv_path","study_instance_uid","series_instance_uid","output_path","status","metrics_json"]
        write_csv(out_root/"pilot_manifest.csv",[],fields)
    pair_fields=list(pair_manifest[0]) if pair_manifest else ["pair_id","patient_id","breast_side","partition","status","cc_uid_pair","mlo_uid_pair","cc_output_path","mlo_output_path","failure_json"]
    write_csv(out_root/"pilot_pairs.csv",pair_manifest,pair_fields)
    summary={
      "built_date":str(date.today()),"config_path":str(config_path.resolve()),"config":config_as_dict(config),
      "sample_policy":raw_cfg.get("sample_policy"),"sample_patient_ids":selected,
      "sample_pair_count":len(sample),"sample_view_count":len(sample)*2,
      "processed_image_count":len(image_rows),"processed_pair_count":sum(r["status"]=="processed" for r in pair_manifest),
      "failed_or_review_required_pairs":failures,"per_image_metrics":[json.loads(r["metrics_json"]) for r in image_rows],
      "stage2_manifest_hashes_unchanged":original_hashes==after_hashes,
      "source_dicom_hashes_unchanged":source_hashes==after_source_hashes,
      "source_dicom_size_mtime_unchanged":source_stats==after_source_stats,
      "no_validation_or_holdout_pixels_read":True,"official_test_images_accessed":False,
      "pixel_values_read_only_for_training_sample":True,
      "contact_sheet_path":str(contact_path.resolve()) if qa_paths else "",
    }
    (out_root/"pilot_run_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return summary


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(r"D:\rp"))
    parser.add_argument("--config",type=Path,default=Path(r"D:\rp\configs\preprocessing\stage3_pilot.json"))
    args=parser.parse_args()
    result=run_pilot(args.root,args.config)
    print(json.dumps({k:result[k] for k in ("sample_patient_ids","sample_pair_count","processed_image_count","processed_pair_count","failed_or_review_required_pairs","stage2_manifest_hashes_unchanged","source_dicom_hashes_unchanged","contact_sheet_path")},indent=2))

if __name__=="__main__":
    main()
