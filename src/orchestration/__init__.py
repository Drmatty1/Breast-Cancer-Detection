"""
Multi-Agent Orchestration Package.
Governed by: DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03).
"""
# 1. Unconditionally available lightweight contract (zero ML dependencies)
from src.orchestration.supervisor_contract import (
    SupervisorState,
    AgentFindingPayload,
    ValidationIssue,
    StateValidationError,
    sanitize_for_json
)
from src.orchestration.supervisor_schema import (
    ConsensusDecision,
    SupervisorSchemaError,
    extract_evidence_quotes,
    validate_supervisor_output_guardrails,
    compute_birads_step_distance,
    evaluate_methodology_deferral_diamond,
    compute_calibrated_confidence_interval
)
from src.orchestration.llm_supervisor import (
    LLMSupervisor,
    BaseLLMClient,
    OllamaLLMClient,
    MockLLMClient,
    build_supervisor_prompt
)

__all__ = [
    "SupervisorState",
    "AgentFindingPayload",
    "ValidationIssue",
    "StateValidationError",
    "sanitize_for_json",
    "ConsensusDecision",
    "SupervisorSchemaError",
    "extract_evidence_quotes",
    "validate_supervisor_output_guardrails",
    "LLMSupervisor",
    "BaseLLMClient",
    "OllamaLLMClient",
    "MockLLMClient",
    "build_supervisor_prompt",
    "MultiAgentOrchestrator",
    "OrchestrationRequest",
    "OrchestrationResponse",
    "OrchestrationAuditRecord",
    "MammoInferenceError",
    "MammoSchemaValidationError",
    "adapt_orchestration_response",
    "adapt_mammo_finding_card",
    "adapt_clin_finding_card",
    "create_unavailable_pathology_payload"
]


def __getattr__(name: str):
    """
    Lazy loader for perception-node orchestration components.
    Ensures importing the lightweight supervisor_contract does not trigger
    heavy perception imports (PyTorch, Torchvision, timm, XGBoost, jsonschema).
    """
    if name in (
        "MultiAgentOrchestrator",
        "OrchestrationRequest",
        "OrchestrationResponse",
        "OrchestrationAuditRecord",
        "MammoInferenceError",
        "MammoSchemaValidationError"
    ):
        from src.orchestration import multi_agent_orchestrator as _mao
        val = getattr(_mao, name)
        globals()[name] = val
        return val

    if name in (
        "adapt_orchestration_response",
        "adapt_mammo_finding_card",
        "adapt_clin_finding_card",
        "create_unavailable_pathology_payload"
    ):
        from src.orchestration import state_adapter as _sa
        val = getattr(_sa, name)
        globals()[name] = val
        return val

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
