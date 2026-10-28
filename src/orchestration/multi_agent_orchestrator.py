"""
Multi-Agent Orchestrator for Mammography and Clinical Risk Agents.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§§ 3.1, 3.4, 5.1, 5.3)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)
- reports/clin/CLIN_AGENT_C3_7_ORCHESTRATION_CONTRACT.md

Key Architecture:
1. Mammo Agent is the sole Authoritative Diagnostic Tier.
2. Clin Agent runs strictly in an isolated Shadow Research Tier.
3. Dual-channel output: returns the exact, unmutated Mammo finding card
   while attaching the Clin finding card only in an independent shadow channel.
4. Hard isolation: Clin outputs cannot modify Mammo predictions, participate
   in consensus voting, or trigger false clinical deferrals.
5. Fail-safe execution: Clin exceptions, timeouts, schema errors, or missing
   covariates never block or mutate the primary Mammo result.
6. Structured audit logging: captures execution provenance, durations, and statuses
   without persisting raw images or sensitive personal health identifiers.
"""
from typing import Dict, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import json
import time
import uuid
import threading
import concurrent.futures

from src.mammo.finding_card import validate_finding_card


class MammoInferenceError(RuntimeError):
    """Raised when the primary authoritative Mammography Agent fails."""
    pass


class MammoSchemaValidationError(ValueError):
    """Raised when the authoritative Mammography finding card violates the schema."""
    pass


@dataclass(frozen=True)
class OrchestrationRequest:
    """
    Typed multimodal request container for the MultiAgentOrchestrator.
    """
    pair_identifier: str
    cc_input: Any
    mlo_input: Any
    clinical_covariates: Optional[Dict[str, Any]] = None
    abnormality_type: Optional[str] = None
    generate_saliency: bool = True
    saliency_output_dir: Optional[Union[str, Path]] = None
    audit_output_dir: Optional[Union[str, Path]] = None


@dataclass(frozen=True)
class OrchestrationAuditRecord:
    """
    Structured, privacy-preserving audit record for multi-agent execution.
    """
    audit_id: str
    pair_identifier: str
    timestamp_utc: str
    orchestrator_version: str
    mammo_status: str          # "SUCCESS", "FAILED", "SCHEMA_INVALID"
    clin_status: str           # "SUCCESS", "ABSTAINED", "FAILED", "TIMEOUT", "SCHEMA_INVALID", "SKIPPED"
    mammo_duration_ms: float
    clin_duration_ms: float
    schema_validation_passed: bool
    clin_error_code: Optional[str] = None
    audit_persisted: bool = False
    logging_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrchestrationResponse:
    """
    Dual-channel response returned by MultiAgentOrchestrator.
    """
    # Primary Authoritative Channel (Exact, unmutated Mammo finding card)
    primary_finding_card: Dict[str, Any]

    # Shadow Research Channel (Clin finding card or None)
    shadow_finding_card: Optional[Dict[str, Any]]

    # Execution Audit Record
    audit_record: OrchestrationAuditRecord

    @property
    def mammo_status(self) -> str:
        return self.audit_record.mammo_status

    @property
    def clin_status(self) -> str:
        return self.audit_record.clin_status

    @property
    def audit_id(self) -> str:
        return self.audit_record.audit_id


class MultiAgentOrchestrator:
    """
    Insulated Multi-Agent Orchestrator.
    Executes authoritative visual inference and secondary shadow research inference.
    """
    VERSION: str = "multi_agent_orchestrator_v1.0"

    def __init__(
        self,
        mammo_pipeline: Any,
        clin_pipeline: Optional[Any] = None,
        audit_logger: Optional[Callable[[Dict[str, Any]], bool]] = None,
        timeout_seconds: float = 5.0,
        max_workers: int = 2
    ):
        if mammo_pipeline is None:
            raise ValueError("MultiAgentOrchestrator requires a valid mammo_pipeline instance.")
        self.mammo_pipeline = mammo_pipeline
        self.clin_pipeline = clin_pipeline
        self.audit_logger = audit_logger
        self.timeout_seconds = timeout_seconds
        self.max_workers = max_workers
        self._thread_prefix = f"clin_shadow_{id(self)}"
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._is_closed = False
        self._capacity_semaphore = threading.BoundedSemaphore(self.max_workers)

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Lazily initializes the managed bounded thread pool executor."""
        with self._lock:
            if self._is_closed:
                raise RuntimeError("Cannot obtain executor on closed MultiAgentOrchestrator.")
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix=self._thread_prefix
                )
            return self._executor

    def close(self, wait: bool = False) -> None:
        """
        Closes the managed shadow executor.
        Idempotent: safe to call repeatedly.
        If wait=False, does not block on running background tasks.
        """
        with self._lock:
            self._is_closed = True
            if self._executor is not None:
                self._executor.shutdown(wait=wait, cancel_futures=True)
                self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(wait=False)

    def __del__(self):
        try:
            self.close(wait=False)
        except Exception:
            pass

    def _execute_clin_shadow(
        self,
        covariates: Optional[Dict[str, Any]],
        pair_id: str
    ) -> Tuple[Optional[Dict[str, Any]], str, float, Optional[str]]:
        """
        Executes Clin Agent in an isolated shadow execution boundary with bounded timeout.
        Returns: (finding_card, clin_status, duration_ms, error_code)
        """
        if self._is_closed:
            return None, "SKIPPED", 0.0, "orchestrator_closed"

        if self.clin_pipeline is None:
            return None, "SKIPPED", 0.0, "clin_pipeline_not_configured"

        # Worker capacity saturation check: if all worker slots are occupied by slow/hanging
        # tasks, do not queue unbounded work. Return SKIPPED immediately to protect the primary
        # diagnostic latency and prevent unbounded queue growth.
        acquired = self._capacity_semaphore.acquire(blocking=False)
        if not acquired:
            return None, "SKIPPED", 0.0, "clin_worker_capacity_saturated"

        # Strict allowlist extraction: extract ONLY breast_density and breast_side
        allowlisted_covariates: Optional[Dict[str, Any]] = None
        if isinstance(covariates, dict):
            allowlisted_covariates = {
                k: covariates[k]
                for k in ("breast_density", "breast_side")
                if k in covariates
            }

        start_time = time.perf_counter()
        clin_card: Optional[Dict[str, Any]] = None
        status = "FAILED"
        error_code: Optional[str] = None

        try:
            executor = self._get_executor()
            future = executor.submit(
                self.clin_pipeline.predict,
                clinical_covariates=allowlisted_covariates,
                pair_identifier=pair_id
            )
            # Release capacity only when worker thread completes actual execution
            future.add_done_callback(lambda _: self._capacity_semaphore.release())
        except Exception as e:
            self._capacity_semaphore.release()
            return None, "FAILED", 0.0, f"clin_submission_failed: {str(e)}"

        try:
            clin_card = future.result(timeout=self.timeout_seconds)

            # Validate returned finding card against schema
            is_valid, val_err = validate_finding_card(clin_card)
            if not is_valid:
                status = "SCHEMA_INVALID"
                error_code = f"schema_validation_failed: {val_err}"
                clin_card = None
            else:
                card_status = clin_card.get("metadata", {}).get("status", "success")
                if card_status == "abstained":
                    status = "ABSTAINED"
                    error_code = clin_card.get("finding", {}).get("abstention_reason")
                else:
                    status = "SUCCESS"

        except concurrent.futures.TimeoutError:
            future.cancel()
            status = "TIMEOUT"
            error_code = f"clin_timeout_exceeded_{self.timeout_seconds}s"
            clin_card = None
        except Exception as e:
            status = "FAILED"
            error_code = f"clin_exception: {type(e).__name__}: {str(e)}"
            clin_card = None

        duration_ms = float(round((time.perf_counter() - start_time) * 1000.0, 2))
        return clin_card, status, duration_ms, error_code

    def run(
        self,
        request: Optional[OrchestrationRequest] = None,
        **kwargs
    ) -> OrchestrationResponse:
        """
        Executes multi-agent orchestration for a paired mammographic study.

        Args:
            request: Typed OrchestrationRequest container (or pass fields as kwargs).

        Returns:
            OrchestrationResponse containing unmutated Mammo card and separate shadow card.
        """
        if self._is_closed:
            raise RuntimeError("MultiAgentOrchestrator is closed and cannot accept new requests.")

        if request is None:
            request = OrchestrationRequest(**kwargs)

        audit_id = str(uuid.uuid4())
        timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        pair_id = request.pair_identifier

        # --- STEP 1: Authoritative Mammo Agent Execution ---
        mammo_start = time.perf_counter()
        try:
            mammo_card = self.mammo_pipeline.predict(
                cc_input=request.cc_input,
                mlo_input=request.mlo_input,
                generate_saliency=request.generate_saliency,
                saliency_output_dir=request.saliency_output_dir,
                pair_identifier=request.pair_identifier,
                clinical_header=request.clinical_covariates,
                abnormality_type=request.abnormality_type
            )
        except Exception as e:
            # Fatal error on visual agent: propagate explicit error, never substitute Clin output
            raise MammoInferenceError(f"Mammo Agent visual inference failed for {pair_id}: {str(e)}") from e

        mammo_duration_ms = float(round((time.perf_counter() - mammo_start) * 1000.0, 2))

        # Validate Mammo Card Schema
        is_valid, mammo_err = validate_finding_card(mammo_card)
        if not is_valid:
            raise MammoSchemaValidationError(f"Mammo Agent card failed schema validation: {mammo_err}")

        # --- STEP 2: Isolated Clin Agent Shadow Execution ---
        clin_card, clin_status, clin_duration_ms, clin_error = self._execute_clin_shadow(
            request.clinical_covariates,
            pair_id
        )

        # --- STEP 3: Consensus Isolation Enforcement ---
        # Assert strict channel separation: Clin output NEVER merges into primary Mammo card
        if clin_card is not None:
            # Hard invariant check: Clin must have voting_eligibility=False
            clin_voting = clin_card.get("metadata", {}).get("voting_eligibility", True)
            if clin_voting is not False:
                # Discard corrupted Clin card
                clin_card = None
                clin_status = "SCHEMA_INVALID"
                clin_error = "clin_violated_zero_voting_eligibility_invariant"

        # --- STEP 4: Multimodal Audit Record Generation & Persistence ---
        audit_persisted = False
        logging_warning: Optional[str] = None

        audit_record = OrchestrationAuditRecord(
            audit_id=audit_id,
            pair_identifier=pair_id,
            timestamp_utc=timestamp_utc,
            orchestrator_version=self.VERSION,
            mammo_status="SUCCESS",
            clin_status=clin_status,
            mammo_duration_ms=mammo_duration_ms,
            clin_duration_ms=clin_duration_ms,
            schema_validation_passed=True,
            clin_error_code=clin_error,
            audit_persisted=False,
            logging_warning=None
        )

        # Attempt audit persistence
        if self.audit_logger is not None:
            try:
                persist_ok = self.audit_logger(audit_record.to_dict())
                if persist_ok:
                    audit_persisted = True
                else:
                    logging_warning = "audit_logger_returned_false"
            except Exception as e:
                # Catch logging failure without corrupting Mammo result
                logging_warning = f"audit_logger_exception: {type(e).__name__}: {str(e)}"
                audit_persisted = False

        final_audit_record = OrchestrationAuditRecord(
            audit_id=audit_id,
            pair_identifier=pair_id,
            timestamp_utc=timestamp_utc,
            orchestrator_version=self.VERSION,
            mammo_status="SUCCESS",
            clin_status=clin_status,
            mammo_duration_ms=mammo_duration_ms,
            clin_duration_ms=clin_duration_ms,
            schema_validation_passed=True,
            clin_error_code=clin_error,
            audit_persisted=audit_persisted,
            logging_warning=logging_warning
        )

        # --- STEP 5: Dual-Channel Output Dispatch ---
        # Return exact, unaltered Mammo card as primary_finding_card
        return OrchestrationResponse(
            primary_finding_card=mammo_card,
            shadow_finding_card=clin_card,
            audit_record=final_audit_record
        )
