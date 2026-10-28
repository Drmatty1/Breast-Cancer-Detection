"""
Unit and Contract Tests for Clin Agent Integration and Multi-Agent Safeguards.

Governed by:
- docs/breast_cancer___ps1___methodology.pdf (§§3.4, 5.1, 5.3)
- DECISIONS.md (DEC-OWNER-01, DEC-OWNER-02, DEC-OWNER-03)
- schemas/finding_card.schema.json

Validates:
1. Legacy Mammo finding card compatibility against the updated schema.
2. Valid Clin Agent finding card acceptance by the schema.
3. Strict nullable lesion handling (string or null allowed; numbers/arrays/objects rejected).
4. Incomplete tabular covariate handling (abstention card generation and schema compliance).
5. Consensus isolation: Clin Agent outputs cannot alter Mammo predictions or trigger false deferrals.
"""
import copy
import json
from pathlib import Path
import unittest
import jsonschema

from src.mammo.finding_card import build_finding_card, validate_finding_card


SCHEMA_PATH = Path(r"D:\rp\schemas\finding_card.schema.json")


def build_clin_finding_card(
    breast_density: int = None,
    breast_side: str = None,
    calibrated_prob: float = None,
    uncalibrated_prob: float = None,
    tree_shap_attributions: dict = None,
    model_version: str = "clin_agent_v1.0"
) -> dict:
    """
    Contract-compliant builder for Clin Agent finding cards.
    Adheres strictly to DEC-OWNER-01, DEC-OWNER-02, and DEC-OWNER-03.
    """
    # Check for missing required tabular covariates
    if breast_density is None or breast_side is None or calibrated_prob is None:
        # Emit an explicit abstention finding card
        return {
            "agent": "clin",
            "finding": {
                "lesion": None,
                "margin": None,
                "calcification": None,
                "risk_profile": "unavailable",
                "abstention_reason": "missing_required_covariates"
            },
            "BI_RADS_a": "unspecified",
            "conf_a": 0.0,
            "sal": {
                "top_regions": [],
                "path": None,
                "cc_path": None,
                "mlo_path": None
            },
            "sh": {
                "breast_density": breast_density,
                "breast_side": breast_side
            },
            "rationale": (
                "Clinical Risk Agent abstained from prediction due to missing "
                "or incomplete required tabular covariates."
            ),
            "metadata": {
                "model_architecture": "xgboost_tabular_logistic",
                "model_version": model_version,
                "status": "abstained",
                "authority": "non_authoritative_research",
                "voting_eligibility": False,
                "is_clinically_validated": False,
                "confidence_provenance": {
                    "conf_a_source": "non_authoritative_zero_weight_convention",
                    "is_conf_a_calibrated": False,
                    "malignancy_calibrated": False
                }
            }
        }

    sal_dict = {
        "top_regions": [],
        "path": None,
        "cc_path": None,
        "mlo_path": None
    }
    if tree_shap_attributions:
        sal_dict["tree_shap"] = tree_shap_attributions

    rationale = (
        f"Tabular clinical model estimated calibrated malignancy risk of {calibrated_prob:.4f}. "
        "Due to negative-result standalone discrimination (ROC-AUC 0.4907), "
        "agent is designated non-authoritative with zero consensus voting weight."
    )

    return {
        "agent": "clin",
        "finding": {
            "lesion": None,
            "margin": None,
            "calcification": None,
            "risk_profile": "malignancy_risk",
            "breast_density": breast_density,
            "breast_side": breast_side
        },
        "BI_RADS_a": "unspecified",
        "conf_a": 0.0,
        "sal": sal_dict,
        "sh": {
            "breast_density": breast_density,
            "breast_side": breast_side
        },
        "rationale": rationale,
        "metadata": {
            "model_architecture": "xgboost_tabular_logistic",
            "model_version": model_version,
            "status": "completed",
            "authority": "non_authoritative_research",
            "voting_eligibility": False,
            "is_clinically_validated": False,
            "task_probabilities": {
                "malignancy_calibrated": float(round(calibrated_prob, 6)),
                "malignancy_uncalibrated": float(round(uncalibrated_prob, 6)) if uncalibrated_prob is not None else None
            },
            "confidence_provenance": {
                "conf_a_source": "non_authoritative_zero_weight_convention",
                "is_conf_a_calibrated": False,
                "malignancy_calibrated": True
            }
        }
    }


def simulate_multiagent_consensus(cards: list) -> dict:
    """
    Reference consensus combinator implementing DEC-OWNER-02 and DEC-OWNER-03.
    Demonstrates consensus isolation:
    - Non-authoritative agents (voting_eligibility == False) are excluded from voting.
    - Ordinal BI-RADS difference deferral check ignores 'unspecified' categories and non-voting agents.
    """
    voting_cards = [
        c for c in cards
        if c.get("metadata", {}).get("voting_eligibility", True) is True
        and c.get("metadata", {}).get("authority") != "non_authoritative_research"
    ]
    shadow_cards = [
        c for c in cards
        if c not in voting_cards
    ]

    if not voting_cards:
        raise ValueError("No eligible voting agents in consensus panel.")

    # In a panel where only Mammo votes:
    primary_card = voting_cards[0]
    consensus_birads = primary_card["BI_RADS_a"]
    consensus_conf = primary_card["conf_a"]

    # Deferral logic per methodology §5.3:
    # Check pairwise BI-RADS rank difference >= 2 ONLY among voting, authoritative agents with numeric ranks
    birads_rank_map = {"2": 2, "3": 3, "4": 4, "4A": 4, "4B": 4, "4C": 4, "5": 5}
    ranks = [
        birads_rank_map[c["BI_RADS_a"]]
        for c in voting_cards
        if c.get("BI_RADS_a") in birads_rank_map
    ]

    deferral_required = False
    if len(ranks) >= 2:
        max_diff = max(ranks) - min(ranks)
        if max_diff >= 2:
            deferral_required = True

    return {
        "consensus_birads": consensus_birads,
        "consensus_conf": consensus_conf,
        "deferral_required": deferral_required,
        "voting_agents": [c["agent"] for c in voting_cards],
        "shadow_agents": [c["agent"] for c in shadow_cards]
    }


class TestClinAgentContract(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCHEMA_PATH.is_file(), f"Missing schema at: {SCHEMA_PATH}")
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

    def test_legacy_mammo_card_compatibility(self):
        """1. Assert authentic Mammography Agent finding card passes validation under the updated schema."""
        mammo_card = build_finding_card(
            binary_prob_calibrated=0.824,
            margin_probs=[0.05, 0.10, 0.05, 0.80],
            calc_probs=[0.10, 0.05, 0.05, 0.75, 0.05],
            birads_ordinal_level=3,
            birads_confidence=0.85,
            saliency_regions=[[120.0, 150.0, 300.0, 350.0]],
            saliency_path="saliency/example_mammo.png",
            abnormality_type="both"
        )
        # Direct jsonschema validation
        jsonschema.validate(instance=mammo_card, schema=self.schema)
        
        # Module validator function
        is_valid, err = validate_finding_card(mammo_card, schema_path=SCHEMA_PATH)
        self.assertTrue(is_valid, f"Legacy Mammo card failed validation: {err}")
        self.assertEqual(mammo_card["agent"], "mammo")
        self.assertEqual(mammo_card["finding"]["lesion"], "malignant")
        self.assertEqual(mammo_card["BI_RADS_a"], "5")

    def test_clin_card_valid_schema_pass(self):
        """2. Assert complete Clin Agent research finding card validates successfully."""
        clin_card = build_clin_finding_card(
            breast_density=2,
            breast_side="LEFT",
            calibrated_prob=0.443079,
            uncalibrated_prob=0.445210,
            tree_shap_attributions={
                "features": ["breast_density", "breast_side_RIGHT"],
                "contributions": [0.0125, -0.0048],
                "bias": -0.2251
            }
        )
        # Verify schema pass
        jsonschema.validate(instance=clin_card, schema=self.schema)
        is_valid, err = validate_finding_card(clin_card, schema_path=SCHEMA_PATH)
        self.assertTrue(is_valid, f"Clin finding card failed validation: {err}")

        # Verify exact governance properties
        self.assertEqual(clin_card["agent"], "clin")
        self.assertIsNone(clin_card["finding"]["lesion"])
        self.assertEqual(clin_card["BI_RADS_a"], "unspecified")
        self.assertEqual(clin_card["conf_a"], 0.0)
        self.assertFalse(clin_card["metadata"]["voting_eligibility"])
        self.assertEqual(clin_card["metadata"]["authority"], "non_authoritative_research")

    def test_nullable_lesion_handling(self):
        """3. Assert lesion accepts valid strings or null, but strictly rejects non-string/non-null types."""
        base_card = build_clin_finding_card(
            breast_density=3,
            breast_side="RIGHT",
            calibrated_prob=0.450
        )

        # Valid string cases
        for valid_val in ["malignant", "benign", "present", "unavailable"]:
            card = copy.deepcopy(base_card)
            card["finding"]["lesion"] = valid_val
            jsonschema.validate(instance=card, schema=self.schema)

        # Valid null case
        card_null = copy.deepcopy(base_card)
        card_null["finding"]["lesion"] = None
        jsonschema.validate(instance=card_null, schema=self.schema)

        # Invalid types: integers, floats, booleans, lists, objects
        invalid_types = [123, 0.45, True, False, ["malignant"], {"status": "present"}]
        for invalid_val in invalid_types:
            card_invalid = copy.deepcopy(base_card)
            card_invalid["finding"]["lesion"] = invalid_val
            with self.assertRaises(jsonschema.ValidationError, msg=f"Failed to reject type: {type(invalid_val)}"):
                jsonschema.validate(instance=card_invalid, schema=self.schema)

    def test_missing_covariates_abstention(self):
        """4. Assert missing tabular covariates produce an explicit abstention card with conf_a=0.0."""
        # Missing density
        card_missing_density = build_clin_finding_card(
            breast_density=None,
            breast_side="LEFT",
            calibrated_prob=0.44
        )
        # Missing side
        card_missing_side = build_clin_finding_card(
            breast_density=2,
            breast_side=None,
            calibrated_prob=0.44
        )
        # Missing probability
        card_missing_prob = build_clin_finding_card(
            breast_density=2,
            breast_side="LEFT",
            calibrated_prob=None
        )

        for card in [card_missing_density, card_missing_side, card_missing_prob]:
            # Must pass schema validation
            jsonschema.validate(instance=card, schema=self.schema)
            is_valid, err = validate_finding_card(card, schema_path=SCHEMA_PATH)
            self.assertTrue(is_valid, f"Abstention card failed schema validation: {err}")

            # Verify contract invariants
            self.assertEqual(card["agent"], "clin")
            self.assertIsNone(card["finding"]["lesion"])
            self.assertEqual(card["finding"]["risk_profile"], "unavailable")
            self.assertEqual(card["BI_RADS_a"], "unspecified")
            self.assertEqual(card["conf_a"], 0.0)
            self.assertEqual(card["metadata"]["status"], "abstained")
            self.assertFalse(card["metadata"]["voting_eligibility"])

    def test_consensus_isolation_mammo_unaffected(self):
        """5. Assert Clin Agent outputs cannot alter Mammo predictions or cause false deferrals."""
        # 1. Verify source independence: src/mammo/inference.py must not depend on clin modules
        mammo_inference_path = Path(r"D:\rp\src\mammo\inference.py")
        self.assertTrue(mammo_inference_path.is_file())
        with open(mammo_inference_path, "r", encoding="utf-8") as f:
            mammo_code = f.read()
        self.assertNotIn("clin", mammo_code.lower().split())
        self.assertNotIn("xgboost", mammo_code.lower())

        # 2. Simulate multi-agent consensus panel with high-confidence Mammo card and Clin card
        mammo_card = build_finding_card(
            binary_prob_calibrated=0.94,
            margin_probs=[0.02, 0.03, 0.05, 0.90],
            calc_probs=None,
            birads_ordinal_level=3,  # Category 5
            birads_confidence=0.92,
            saliency_regions=[[50.0, 50.0, 200.0, 200.0]],
            abnormality_type="mass"
        )
        clin_card = build_clin_finding_card(
            breast_density=3,
            breast_side="RIGHT",
            calibrated_prob=0.443079,
            uncalibrated_prob=0.445210
        )

        # Baseline: Mammo-only decision
        mammo_only_result = simulate_multiagent_consensus([mammo_card])
        self.assertEqual(mammo_only_result["consensus_birads"], "5")
        self.assertEqual(mammo_only_result["consensus_conf"], 0.92)
        self.assertFalse(mammo_only_result["deferral_required"])

        # Multi-agent panel: Mammo + Clin
        multiagent_result = simulate_multiagent_consensus([mammo_card, clin_card])

        # Assert zero alteration of visual decision
        self.assertEqual(multiagent_result["consensus_birads"], mammo_only_result["consensus_birads"])
        self.assertEqual(multiagent_result["consensus_conf"], mammo_only_result["consensus_conf"])
        # Assert no false deferral triggered by Clin Agent's unspecified BI-RADS or risk
        self.assertFalse(multiagent_result["deferral_required"])
        # Assert proper agent role classification
        self.assertEqual(multiagent_result["voting_agents"], ["mammo"])
        self.assertEqual(multiagent_result["shadow_agents"], ["clin"])


if __name__ == "__main__":
    unittest.main()
