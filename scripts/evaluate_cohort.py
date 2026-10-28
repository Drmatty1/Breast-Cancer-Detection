"""
Batch Cohort Evaluation Script for RP2 LLM Supervisor.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§§ 4.2, 5.1, 5.3, 6, 9.1)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)

Usage:
    python scripts/evaluate_cohort.py --num-cases 10 --model qwen2.5:3b
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
import sys
import csv
import json
import time
import argparse
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration.supervisor_contract import (
    SupervisorState,
    AgentFindingPayload
)
from src.orchestration.supervisor_schema import (
    ConsensusDecision,
    extract_evidence_quotes
)
from src.orchestration.llm_supervisor import (
    LLMSupervisor,
    OllamaLLMClient
)
from scripts.run_case_inference import (
    load_cached_images,
    lookup_clinical_covariates,
    get_agent_finding_cards
)


def compute_roc_auc(y_true: List[int], y_score: List[float]) -> float:
    """Computes ROC-AUC via rank-sum (Wilcoxon-Mann-Whitney)."""
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.argsort(np.argsort(y_score)) + 1
    rank_pos = np.sum(ranks[y_true == 1])
    u = rank_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def run_cohort_evaluation(
    num_cases: int = 10,
    model_name: str = "qwen2.5:3b",
    theta: float = 0.0,
    manifest_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
):
    output_dir = output_dir or PROJECT_ROOT / "reports" / "rp2_experimental" / "cohort_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_path or PROJECT_ROOT / "data" / "manifests" / "clin_feature_table.csv"
    data_dir = PROJECT_ROOT / "data" / "cached_preprocessed_512x512"

    print("=" * 75)
    print(f"RP2 Mammography Agent: Running Cohort Batch Evaluation")
    print(f"Model: {model_name} | Target Sample Count: {num_cases} | Uncertainty Theta: {theta}")
    print(f"Manifest: {manifest_path.name}")
    print("=" * 75)

    # Collect available cases that have both CC and MLO cached arrays
    eligible_cases = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["pair_id"]
            cc_file = data_dir / f"{pid}_CC.npy"
            mlo_file = data_dir / f"{pid}_MLO.npy"
            if cc_file.is_file() and mlo_file.is_file():
                eligible_cases.append(pid)

    print(f"\nDiscovered {len(eligible_cases)} fully preprocessed image pairs with clinical metadata.")
    selected_cases = eligible_cases[:num_cases]
    print(f"Evaluating {len(selected_cases)} selected cases...\n")

    client = OllamaLLMClient(model_name=model_name)
    supervisor = LLMSupervisor(client=client, theta=theta)

    results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    guardrail_passes = 0
    total_quotes_found = 0
    deferrals = 0

    print(f"{'Case ID':<16} | {'Target':<9} | {'BI-RADS':<7} | {'p_malig':<7} | {'Quotes':<6} | {'Defer?':<6} | {'Status':<10} | {'Latency'}")
    print("-" * 75)

    start_eval_time = time.time()
    for idx, case_id in enumerate(selected_cases, 1):
        try:
            img_meta = load_cached_images(case_id, data_dir)
            covariates = lookup_clinical_covariates(case_id, manifest_path)
            mammo_card, clin_card = get_agent_finding_cards(case_id, covariates, img_meta)

            mammo_payload = AgentFindingPayload(
                agent_id="mammo",
                status="success",
                authority_level="authoritative",
                voting_eligible=True,
                bi_rads=mammo_card["BI_RADS_a"],
                confidence=mammo_card["conf_a"],
                confidence_type=mammo_card["confidence_type"],
                p_malignancy=mammo_card["conf_a"],
                findings_summary=mammo_card["finding"],
                task_probabilities=mammo_card["metadata"]["task_probabilities"],
                explainability=mammo_card["sal"],
                rationale=mammo_card["rationale"],
                raw_finding_card=mammo_card,
                metadata={}
            )
            clin_payload = AgentFindingPayload(
                agent_id="clin",
                status="success",
                authority_level="non_authoritative_research",
                voting_eligible=False,
                bi_rads=clin_card["BI_RADS_a"],
                confidence=clin_card["conf_a"],
                confidence_type=clin_card["confidence_type"],
                p_malignancy=clin_card["metadata"]["task_probabilities"]["malignancy_calibrated"],
                findings_summary=clin_card["finding"],
                task_probabilities=clin_card["metadata"]["task_probabilities"],
                explainability={"tree_shap_attributions": clin_card["sh"]},
                rationale=clin_card["rationale"],
                raw_finding_card=clin_card,
                metadata={}
            )

            state = SupervisorState(
                study_id=case_id,
                timestamp_utc="2026-10-27T12:00:00Z",
                agents={"mammo": mammo_payload, "clin": clin_payload},
                defer_to_human=False
            )

            decision = supervisor.deliberate(state)
            quotes = extract_evidence_quotes(decision.evidence_rationale)
            quote_count = len(quotes)
            total_quotes_found += quote_count

            status = decision.audit_metadata.get("guardrail_status", "unknown")
            if status == "passed":
                guardrail_passes += 1

            if decision.defer_to_human:
                deferrals += 1

            lat = float(decision.audit_metadata.get("latency_ms", 0.0))
            latencies.append(lat)

            print(f"{case_id:<16} | {covariates['target_name']:<9} | {decision.consensus_bi_rads:<7} | {decision.malignancy_probability:<7.4f} | {quote_count:<6} | {str(decision.defer_to_human):<6} | {status:<10} | {lat:.0f} ms")

            results.append({
                "case_id": case_id,
                "target": covariates["target_name"],
                "target_binary": covariates["target_binary"],
                "density": covariates["breast_density"],
                "abnormality": covariates["abnormality_type"],
                "consensus_bi_rads": decision.consensus_bi_rads,
                "malignancy_probability": decision.malignancy_probability,
                "confidence_interval": decision.confidence_interval,
                "defer_to_human": decision.defer_to_human,
                "guardrail_status": status,
                "quotes_count": quote_count,
                "quotes": quotes,
                "latency_ms": lat,
                "rationale": decision.evidence_rationale
            })

        except Exception as e:
            print(f"{case_id:<16} | ERROR: {e}")

    total_eval_duration = time.time() - start_eval_time
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    guardrail_pass_rate = (guardrail_passes / len(selected_cases) * 100) if selected_cases else 0.0
    deferral_rate = (deferrals / len(selected_cases) * 100) if selected_cases else 0.0
    avg_quotes = total_quotes_found / len(selected_cases) if selected_cases else 0.0

    # Diagnostic & selective classification metrics comparing predictions against real answer
    y_true_all = [r["target_binary"] for r in results]
    p_all = [r["malignancy_probability"] for r in results]
    tau = 0.367477
    y_pred_all = [1 if p >= tau else 0 for p in p_all]

    def calc_stats(y_t, y_p):
        y_t, y_p = np.array(y_t), np.array(y_p)
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        acc = (tp + tn) / len(y_t) if len(y_t) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": acc, "sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "f1": f1}

    stats_all = calc_stats(y_true_all, y_pred_all)
    auc_all = compute_roc_auc(y_true_all, p_all)

    # Autonomous retained cohort (cases not deferred)
    retained_results = [r for r in results if not r["defer_to_human"]]
    stats_ret = None
    auc_ret = None
    if retained_results:
        y_true_ret = [r["target_binary"] for r in retained_results]
        p_ret = [r["malignancy_probability"] for r in retained_results]
        y_pred_ret = [1 if p >= tau else 0 for p in p_ret]
        stats_ret = calc_stats(y_true_ret, y_pred_ret)
        auc_ret = compute_roc_auc(y_true_ret, p_ret)

    print("-" * 75)
    print("\n" + "=" * 75)
    print("COHORT EVALUATION METRICS SUMMARY (§9)")
    print("=" * 75)
    print(f"Total Cases Evaluated       : {len(selected_cases)}")
    print(f"Total Execution Time        : {total_eval_duration:.2f} s")
    print(f"Average Deliberation Latency: {avg_latency:.1f} ms/case")
    print(f"Guardrail Pass Rate         : {guardrail_pass_rate:.1f}%")
    print(f"Human Deferral Rate         : {deferral_rate:.1f}% ({deferrals}/{len(selected_cases)})")
    print(f"Autonomous Coverage Rate    : {100.0 - deferral_rate:.1f}% ({len(retained_results)}/{len(selected_cases)})")
    print(f"Mean Citations Per Case     : {avg_quotes:.1f} quotes/rationale")
    print(f"Hallucination Violations    : {len(selected_cases) - guardrail_passes}")
    print("-" * 75)
    print("DIAGNOSTIC PERFORMANCE (Compared Against Ground Truth Labels):")
    print(f"Decision Threshold (tau)    : {tau:.6f}")
    print(f"Screening Sensitivity       : {stats_all['sens'] * 100:.2f}% ({stats_all['tp']}/{stats_all['tp'] + stats_all['fn']})")
    print(f"Screening Specificity       : {stats_all['spec'] * 100:.2f}% ({stats_all['tn']}/{stats_all['tn'] + stats_all['fp']})")
    print(f"Overall Accuracy            : {stats_all['acc'] * 100:.2f}% ({stats_all['tp'] + stats_all['tn']}/{len(selected_cases)})")
    print(f"Positive Predictive Value   : {stats_all['ppv'] * 100:.2f}%")
    print(f"Negative Predictive Value   : {stats_all['npv'] * 100:.2f}%")
    print(f"F1-Score                    : {stats_all['f1']:.4f}")
    print(f"ROC-AUC                     : {auc_all:.4f}")
    print(f"Confusion Matrix [TP,FP,TN,FN]: TP={stats_all['tp']}, FP={stats_all['fp']}, TN={stats_all['tn']}, FN={stats_all['fn']}")
    if stats_ret:
        print("-" * 75)
        print("AUTONOMOUS RETAINED COHORT (After Uncertainty Deferral):")
        print(f"Retained Cases              : {len(retained_results)}")
        print(f"Retained Sensitivity        : {stats_ret['sens'] * 100:.2f}% ({stats_ret['tp']}/{stats_ret['tp'] + stats_ret['fn']})")
        print(f"Retained Specificity        : {stats_ret['spec'] * 100:.2f}% ({stats_ret['tn']}/{stats_ret['tn'] + stats_ret['fp']})")
        print(f"Retained Accuracy           : {stats_ret['acc'] * 100:.2f}%")
        print(f"Retained ROC-AUC            : {auc_ret:.4f}")
    print("=" * 75)

    # Save summary report
    summary_path = output_dir / f"cohort_evaluation_summary_{len(selected_cases)}cases.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "manifest": str(manifest_path),
            "evaluated_cases": len(selected_cases),
            "guardrail_pass_rate": guardrail_pass_rate,
            "hallucination_violations": len(selected_cases) - guardrail_passes,
            "human_deferral_rate": deferral_rate,
            "autonomous_coverage_rate": 100.0 - deferral_rate,
            "average_latency_ms": avg_latency,
            "mean_citations_per_case": avg_quotes,
            "confidence_interval_method": "calibrated_wilson_80_ci_n50",
            "ground_truth_isolated": True,
            "diagnostic_metrics": {
                "decision_threshold": tau,
                "sensitivity": stats_all["sens"],
                "specificity": stats_all["spec"],
                "accuracy": stats_all["acc"],
                "ppv": stats_all["ppv"],
                "npv": stats_all["npv"],
                "f1": stats_all["f1"],
                "roc_auc": auc_all,
                "confusion_matrix": stats_all
            },
            "autonomous_retained_metrics": {
                "retained_count": len(retained_results),
                "sensitivity": stats_ret["sens"],
                "specificity": stats_ret["spec"],
                "accuracy": stats_ret["acc"],
                "roc_auc": auc_ret,
                "confusion_matrix": stats_ret
            } if stats_ret else None,
            "results": results
        }, f, indent=2)

    print(f"\nCohort evaluation saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="RP2 Cohort Evaluation Battery")
    parser.add_argument("--num-cases", type=int, default=10, help="Number of cases to evaluate from manifest")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="LLM model name in Ollama")
    parser.add_argument("--theta", type=float, default=0.0, help="Uncertainty threshold theta for Methodology Figure 1 Decision Diamond")
    parser.add_argument("--manifest", type=str, default=None, help="Path to clinical feature manifest")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    run_cohort_evaluation(
        num_cases=args.num_cases,
        model_name=args.model,
        theta=args.theta,
        manifest_path=manifest_path,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
