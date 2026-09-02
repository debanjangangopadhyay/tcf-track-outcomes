"""TCF-001 TRACK clinical logic engine.

The engine preserves the original project concepts while making the mathematical
assumptions explicit and separating baseline utility from on-treatment molecular
updates. No new third-party dependency is required beyond the original runtime.
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


ESCAT_POINTS = {
    "Tier I": 100.0,
    "Tier II": 75.0,
    "Tier III": 50.0,
    "Tier IV": 25.0,
}

# The original UI supports these disease groups. This remains intentionally
# transparent rather than pretending to be a complete ESCAT implementation.
KRAS_TIER_I = {"LUAD", "LUSC", "COAD", "READ"}
KRAS_TIER_II = {"PAAD", "CHOL"}


def _as_bool_yes(value: Any) -> bool:
    return str(value if value is not None else "No").strip().title() == "Yes"


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def assign_escat_tier(row: pd.Series) -> str:
    """Transparent, project-specific actionability mapping for the supported cohort.

    IMPORTANT: this is an ESCAT-inspired mapping for the supported KRAS cohorts,
    not a complete implementation of the ESMO ESCAT framework.
    """
    kras = _as_bool_yes(row.get("KRAS_Mutant", "No"))
    tp53 = _as_bool_yes(row.get("TP53_Mutant", "No"))
    cohort = str(row.get("Cohort", "")).strip().upper()

    if kras:
        if cohort in KRAS_TIER_I:
            return "Tier I"
        if cohort in KRAS_TIER_II:
            return "Tier II"
        return "Tier III"

    # Retained from the original project logic.
    if tp53:
        return "Tier IV"
    return "Tier IV"


def calculate_deritis(ast: Any, alt: Any) -> float:
    ast_f = _safe_float(ast, 25.0)
    alt_f = _safe_float(alt, 25.0)
    return ast_f / alt_f if alt_f > 0 else 1.0


def host_toxicity_components(row: pd.Series) -> Dict[str, float]:
    """Return interpretable continuous host/treatment modifiers.

    Unlike the original threshold-only implementation, continuous bounded
    transforms are used so the contribution is smoother and less brittle.
    """
    sii = _safe_float(row.get("SII", 500.0), 500.0)
    age = _safe_float(row.get("Age", 50.0), 50.0)
    de_ritis = calculate_deritis(row.get("AST", 25.0), row.get("ALT", 25.0))

    # Robust, bounded transforms. They are deliberately not presented as
    # disease-specific clinical cut-points.
    sii_stress = float(np.clip((sii - 500.0) / 1000.0, -0.5, 1.0))
    liver_stress = float(np.clip((de_ritis - 1.0) / 2.0, -0.5, 1.0))
    age_stress = float(np.clip((age - 50.0) / 40.0, -0.5, 1.0))

    # Host resilience is an interpretable multiplicative term.
    phi_host = float(np.exp(-0.18 * max(0.0, sii_stress) - 0.18 * max(0.0, liver_stress)))
    tau_tox = float(0.08 * max(0.0, age_stress) + 0.18 * max(0.0, liver_stress))
    toxicity_modifier = float(np.exp(-tau_tox))

    return {
        "SII_Stress": sii_stress,
        "DeRitis": de_ritis,
        "Liver_Stress": liver_stress,
        "Age_Stress": age_stress,
        "Phi_Host": phi_host,
        "Tau_Tox": tau_tox,
        "Toxicity_Modifier": toxicity_modifier,
    }


def ctDNA_kinetics(row: pd.Series) -> Dict[str, float]:
    """Calculate an interpretable baseline-to-follow-up molecular update.

    The original score used a binary/tri-state penalty. We retain its intent but
    expose the actual log-ratio and continuous response fraction.
    """
    f_base = _safe_float(row.get("Tumor_Fraction", 0.05), 0.05)
    f_follow = _safe_float(row.get("Tumor_Fraction_Followup", np.nan), np.nan)

    if not np.isfinite(f_base) or f_base <= 0 or not np.isfinite(f_follow) or f_follow <= 0:
        return {
            "ctDNA_Log_Ratio": np.nan,
            "ctDNA_Relative_Change": np.nan,
            "K_Clearance": 1.0,
            "K_Observed": 0.0,
            "K_Status": "Not available",
        }

    log_ratio = float(np.log(f_base / f_follow))
    relative_change = float(1.0 - (f_follow / f_base))

    # Smooth bounded dynamic update centered on no molecular change.
    # Positive log-ratio = clearance; negative = molecular expansion.
    k_clearance = float(np.clip(0.70 + 0.30 * np.tanh(log_ratio), 0.40, 1.00))
    if log_ratio > 0.693:
        status = "Rapid clearance"
    elif log_ratio > 0:
        status = "Clearance"
    elif log_ratio < -0.05:
        status = "Molecular progression"
    else:
        status = "Stable / low change"

    return {
        "ctDNA_Log_Ratio": log_ratio,
        "ctDNA_Relative_Change": relative_change,
        "K_Clearance": k_clearance,
        "K_Observed": 1.0,
        "K_Status": status,
    }


def calculate_interaction_terms(row: pd.Series) -> Dict[str, float]:
    """Explicit co-mutation interaction features.

    This does not claim biological epistasis. The term is an interaction feature
    that can be estimated empirically in the cohort-level Cox model.
    """
    kras = 1.0 if _as_bool_yes(row.get("KRAS_Mutant", "No")) else 0.0
    tp53 = 1.0 if _as_bool_yes(row.get("TP53_Mutant", "No")) else 0.0
    matched = 1.0 if str(row.get("Therapy_Type", "Standard Care")).strip() == "Targeted/Matched" else 0.0

    return {
        "KRAS_x_TP53": kras * tp53,
        "Matched_x_TP53": matched * tp53,
        "Matched_x_KRAS": matched * kras,
    }


def derive_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create the transparent multidimensional feature state used by CUI."""
    out = df.copy()
    host = out.apply(host_toxicity_components, axis=1, result_type="expand")
    kin = out.apply(ctDNA_kinetics, axis=1, result_type="expand")
    inter = out.apply(calculate_interaction_terms, axis=1, result_type="expand")
    out = pd.concat([out, host, kin, inter], axis=1)

    out["Actionability_Points"] = out["ESCAT_Tier"].map(ESCAT_POINTS).fillna(25.0)
    out["Therapy_Match"] = np.where(out["Therapy_Type"].eq("Targeted/Matched"), 1.0, 0.30)
    out["TP53_Resistance_Base"] = np.where(out["TP53_Mutant"].eq("Yes"), 0.85, 1.0)

    # Baseline composite does not require follow-up ctDNA. The dynamic score can
    # update later when a follow-up measurement exists.
    out["Baseline_Utility_Raw"] = (
        out["Actionability_Points"]
        * out["Therapy_Match"]
        * out["TP53_Resistance_Base"]
        * out["Phi_Host"]
        * out["Toxicity_Modifier"]
    )

    out["Dynamic_Utility_Raw"] = out["Baseline_Utility_Raw"] * out["K_Clearance"]
    out["VMTB_Matching_Score"] = np.clip(out["Dynamic_Utility_Raw"], 0.0, 100.0).round(1)
    out["Baseline_CUI"] = np.clip(out["Baseline_Utility_Raw"], 0.0, 100.0).round(1)
    out["Dynamic_CUI"] = np.clip(out["Dynamic_Utility_Raw"], 0.0, 100.0).round(1)
    return out


def _safe_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=series.index, dtype=float)
    return (x - x.mean()) / sd


def build_cox_features(df: pd.DataFrame, include_kinetics: bool = False) -> pd.DataFrame:
    """Build a transparent baseline or dynamic Cox design matrix."""
    x = pd.DataFrame(index=df.index)
    x["Is_Matched"] = (df["Therapy_Type"] == "Targeted/Matched").astype(float)
    x["KRAS_Mut"] = (df["KRAS_Mutant"] == "Yes").astype(float)
    x["TP53_Mut"] = (df["TP53_Mutant"] == "Yes").astype(float)
    x["KRAS_x_TP53"] = x["KRAS_Mut"] * x["TP53_Mut"]
    x["Matched_x_TP53"] = x["Is_Matched"] * x["TP53_Mut"]
    x["SII_z"] = _safe_z(df["SII"])
    x["DeRitis_z"] = _safe_z(df["DeRitis"] if "DeRitis" in df.columns else pd.Series(np.nan, index=df.index))
    x["Age_z"] = _safe_z(df["Age"])
    if include_kinetics:
        x["ctDNA_Log_Ratio"] = pd.to_numeric(df["ctDNA_Log_Ratio"], errors="coerce").fillna(0.0)
        x["K_Clearance"] = pd.to_numeric(df["K_Clearance"], errors="coerce").fillna(1.0)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_empirical_cui(
    df: pd.DataFrame,
    hazard_ratios: Optional[pd.Series] = None,
    baseline: float = 50.0,
) -> pd.DataFrame:
    """Calibrate the CUI using empirical Cox effects when available.

    The result stays bounded 0-100. When no stable model is available, the
    transparent biological composite remains the fallback rather than silently
    fitting an unstable model.
    """
    out = df.copy()
    if hazard_ratios is None or len(hazard_ratios) == 0:
        out["Calibrated_CUI"] = out["Dynamic_CUI"].astype(float)
        out["CUI_Calibration"] = "Mechanistic fallback (no stable Cox coefficients)"
        return out

    effect = pd.Series(0.0, index=out.index)
    feature_map = {
        "Is_Matched": "Is_Matched",
        "KRAS_Mut": "KRAS_Mut",
        "TP53_Mut": "TP53_Mut",
        "KRAS_x_TP53": "KRAS_x_TP53",
        "Matched_x_TP53": "Matched_x_TP53",
        "SII_z": "SII_z",
        "DeRitis_z": "DeRitis_z",
        "Age_z": "Age_z",
        "ctDNA_Log_Ratio": "ctDNA_Log_Ratio",
    }
    design = build_cox_features(out, include_kinetics=True)
    available = []
    for name, col in feature_map.items():
        if name in hazard_ratios.index and col in design:
            coef = _safe_float(hazard_ratios.loc[name], 0.0)
            effect += coef * design[col]
            available.append(name)

    # Lower estimated hazard maps to higher utility. Logistic transform prevents
    # exploding scores and makes the score comparable on a 0-100 scale.
    risk_component = 1.0 / (1.0 + np.exp(np.clip(effect, -20, 20)))
    out["Empirical_Risk_Component"] = risk_component
    out["Calibrated_CUI"] = np.clip(100.0 * (0.35 * out["Dynamic_CUI"] / 100.0 + 0.65 * risk_component), 0, 100).round(1)
    out["CUI_Calibration"] = "Empirical Cox + mechanistic hybrid: " + ", ".join(available)
    return out


def score_explanation(row: pd.Series) -> Dict[str, float]:
    """Return component values for transparent bedside-style explanation."""
    return {
        "Actionability": float(row.get("Actionability_Points", 25.0)),
        "Therapy Match": float(row.get("Therapy_Match", 0.30)),
        "Resistance Modifier": float(row.get("TP53_Resistance_Base", 1.0)),
        "Host Resilience": float(row.get("Phi_Host", 1.0)),
        "Toxicity Modifier": float(row.get("Toxicity_Modifier", 1.0)),
        "Kinetic Modifier": float(row.get("K_Clearance", 1.0)),
        "Baseline CUI": float(row.get("Baseline_CUI", 0.0)),
        "Dynamic CUI": float(row.get("Dynamic_CUI", 0.0)),
        "Calibrated CUI": float(row.get("Calibrated_CUI", row.get("Dynamic_CUI", 0.0))),
    }
    
