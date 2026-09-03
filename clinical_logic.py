"""TCF-001 TRACK clinical logic engine."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MODEL_ID = "TCF-001-TRACK"
ENGINE_ID = "TC-KUO"
ARCHITECTURE_VERSION = "3.0-research"
MODEL_STATUS = "RESEARCH_USE_ONLY"
OUTPUT_SEMANTICS = "BOUNDED_UTILITY_INDEX_NOT_PROBABILITY"

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny


def safe_float(value: Any, default: float = np.nan, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(x):
        return default
    if minimum is not None and x < minimum:
        return default
    if maximum is not None and x > maximum:
        return default
    return x


def bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not np.isfinite(value):
        return low
    return float(np.clip(value, low, high))


def stable_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safe_log_fraction(value: Any) -> float:
    x = safe_float(value, default=np.nan, minimum=TINY, maximum=1.0)
    if not np.isfinite(x):
        return np.nan
    return float(math.log(x))


def geometric_mean(values: Sequence[float], neutral: float = 1.0) -> float:
    vals = [float(v) for v in values if np.isfinite(v) and v > 0]
    if not vals:
        return neutral
    return float(math.exp(np.mean(np.log(vals))))


@dataclass(frozen=True)
class EvidenceParameters:
    tier_points: Mapping[str, float] = field(
        default_factory=lambda: {
            "Tier I": 100.0, "Tier II": 75.0, "Tier III": 50.0,
            "Tier IV": 25.0, "Unknown": 0.0,
        }
    )
    confidence_floor: float = 0.25


@dataclass(frozen=True)
class GenomicTensorParameters:
    beta_kras: float = 0.00
    beta_tp53: float = 0.00
    beta_therapy: float = 0.50
    beta_kras_tp53: float = -0.35
    beta_kras_therapy: float = 0.20
    beta_tp53_therapy: float = 0.25
    beta_kras_tp53_therapy: float = -0.15
    intercept: float = 0.0
    sigmoid_temperature: float = 1.0


@dataclass(frozen=True)
class HostFieldParameters:
    sii_reference: float = 500.0
    sii_scale: float = 500.0
    sii_exponent: float = 1.5
    sii_weight: float = 0.15
    deritis_reference: float = 1.0
    deritis_scale: float = 1.0
    deritis_exponent: float = 4.0 / 3.0
    deritis_weight: float = 0.25
    age_reference: float = 50.0
    age_scale: float = 40.0
    age_exponent: float = 1.0
    age_weight: float = 0.10
    missingness_penalty: float = 0.05


@dataclass(frozen=True)
class KineticParameters:
    velocity_slope: float = 1.50
    velocity_lower_bound: float = 0.40
    velocity_range: float = 0.60
    acceleration_slope: float = 0.75
    acceleration_weight: float = 0.15
    rebound_weight: float = 0.20
    plateau_tolerance: float = 0.10
    minimum_points_for_regression: int = 2
    preferred_points_for_acceleration: int = 3


@dataclass(frozen=True)
class QualityParameters:
    default_quality: float = 0.50
    minimum_quality: float = 0.0
    maximum_quality: float = 1.0
    kinetic_quality_floor: float = 0.25


@dataclass(frozen=True)
class UtilityParameters:
    evidence_weight: float = 1.0
    host_weight: float = 1.0
    treatment_weight: float = 1.0
    kinetic_weight: float = 1.0
    utility_scale: float = 65.0
    dynamic_quality_weight: float = 0.25


@dataclass(frozen=True)
class TCFTParameters:
    evidence: EvidenceParameters = field(default_factory=EvidenceParameters)
    genomic: GenomicTensorParameters = field(default_factory=GenomicTensorParameters)
    host: HostFieldParameters = field(default_factory=HostFieldParameters)
    kinetics: KineticParameters = field(default_factory=KineticParameters)
    quality: QualityParameters = field(default_factory=QualityParameters)
    utility: UtilityParameters = field(default_factory=UtilityParameters)


DEFAULT_PARAMETERS = TCFTParameters()


def parameter_fingerprint(parameters: TCFTParameters = DEFAULT_PARAMETERS) -> str:
    payload = json.dumps(asdict(parameters), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evidence_points(evidence_tier: Any, parameters: TCFTParameters = DEFAULT_PARAMETERS) -> Tuple[float, bool]:
    tier = str(evidence_tier if evidence_tier is not None else "Unknown").strip()
    points = parameters.evidence.tier_points.get(tier, parameters.evidence.tier_points["Unknown"])
    observed = tier != "Unknown"
    return float(points), observed


def normalized_evidence(evidence_tier: Any, evidence_confidence: Any = np.nan, parameters: TCFTParameters = DEFAULT_PARAMETERS) -> Dict[str, float]:
    points, observed = evidence_points(evidence_tier, parameters)
    confidence = safe_float(evidence_confidence, default=np.nan, minimum=0.0, maximum=1.0)
    if not np.isfinite(confidence):
        confidence = 1.0 if observed else parameters.evidence.confidence_floor
    strength = bounded(points / 100.0)
    return {"Evidence_Strength": strength, "Evidence_Confidence": confidence, "Evidence_Observed": float(observed)}


def encode_variant_state(observed: Any, allele_fraction: Any = np.nan, clonality: Any = np.nan) -> Dict[str, float]:
    present = float(bool(observed))
    af = safe_float(allele_fraction, default=np.nan, minimum=0.0, maximum=1.0)
    clonal = safe_float(clonality, default=np.nan, minimum=0.0, maximum=1.0)
    return {"Variant_State": present, "Allele_Fraction": af, "Clonality": clonal, "Variant_Observed": present}


def therapy_compatibility(match_type: Any, curated_strength: Any = np.nan) -> Dict[str, float]:
    category = str(match_type if match_type is not None else "Unknown").strip().casefold()
    categorical = {
        "matched": 1.00, "targeted/matched": 1.00, "targeted": 0.90,
        "partial": 0.60, "partially_matched": 0.60, "unmatched": 0.00, "unknown": np.nan,
    }
    base = categorical.get(category, np.nan)
    explicit = safe_float(curated_strength, default=np.nan, minimum=0.0, maximum=1.0)

    if np.isfinite(explicit):
        score, observed = explicit, True
    elif np.isfinite(base):
        score, observed = float(base), category != "unknown"
    else:
        score, observed = 0.0, False
    return {"Therapy_Compatibility": score, "Therapy_Compatibility_Observed": float(observed)}


def genomic_interaction_tensor(kras_state: float, tp53_state: float, therapy_state: float, parameters: TCFTParameters = DEFAULT_PARAMETERS) -> Dict[str, float]:
    p = parameters.genomic
    K, P, M = bounded(kras_state), bounded(tp53_state), bounded(therapy_state)
    first_order = p.beta_kras * K + p.beta_tp53 * P + p.beta_therapy * M
    pairwise = p.beta_kras_tp53 * K * P + p.beta_kras_therapy * K * M + p.beta_tp53_therapy * P * M
    third_order = p.beta_kras_tp53_therapy * K * P * M
    raw = p.intercept + first_order + pairwise + third_order
    activated = stable_sigmoid(raw / max(p.sigmoid_temperature, EPS))
    return {
        "Tensor_First_Order": float(first_order), "Tensor_Pairwise_Order": float(pairwise),
        "Tensor_Third_Order": float(third_order), "Genomic_Tensor_Raw": float(raw),
        "Genomic_Interaction_State": float(activated), "KRAS_State": K, "TP53_State": P, "Therapy_State": M
    }


def calculate_deritis(ast: Any, alt: Any) -> Tuple[float, bool]:
    ast_f, alt_f = safe_float(ast, default=np.nan, minimum=0.0), safe_float(alt, default=np.nan, minimum=0.0)
    if not np.isfinite(ast_f) or not np.isfinite(alt_f) or alt_f <= 0:
        return np.nan, False
    return float(ast_f / alt_f), True


def _power_stress(value: float, reference: float, scale: float, exponent: float) -> float:
    if not np.isfinite(value):
        return 0.0
    normalized = max(0.0, (value - reference) / max(scale, EPS))
    return float(normalized ** exponent)


def host_state_field(sii: Any, deritis: Any, age: Any, parameters: TCFTParameters = DEFAULT_PARAMETERS) -> Dict[str, float]:
    p = parameters.host
    sii_f, dr_f, age_f = safe_float(sii, default=np.nan, minimum=0.0), safe_float(deritis, default=np.nan, minimum=0.0), safe_float(age, default=np.nan, minimum=0.0, maximum=120.0)
    sii_stress = _power_stress(sii_f, p.sii_reference, p.sii_scale, p.sii_exponent)
    dr_stress = _power_stress(dr_f, p.deritis_reference, p.deritis_scale, p.deritis_exponent)
    age_stress = _power_stress(age_f, p.age_reference, p.age_scale, p.age_exponent)
    
    sii_modifier = math.exp(-p.sii_weight * sii_stress)
    hepatic_modifier = math.exp(-p.deritis_weight * dr_stress)
    age_modifier = math.exp(-p.age_weight * age_stress)
    
    observed_fraction = float(np.mean([np.isfinite(sii_f), np.isfinite(dr_f), np.isfinite(age_f)]))
    missingness_modifier = 1.0 - p.missingness_penalty * (1.0 - observed_fraction)
    host_field = sii_modifier * hepatic_modifier * age_modifier * missingness_modifier
    
    return {
        "SII_Stress": sii_stress, "DeRitis_Stress": dr_stress, "Age_Stress": age_stress,
        "SII_Modifier": float(sii_modifier), "Hepatic_Modifier": float(hepatic_modifier),
        "Age_Modifier": float(age_modifier), "Host_Missingness_Modifier": float(missingness_modifier),
        "Host_Data_Completeness": observed_fraction, "Host_State_Field": float(np.clip(host_field, 0.0, 1.0))
    }


def assay_quality_state(assay_quality: Any = np.nan, lod_margin: Any = np.nan, replicate_confidence: Any = np.nan, parameters: TCFTParameters = DEFAULT_PARAMETERS) -> Dict[str, float]:
    p = parameters.quality
    values = [safe_float(v, default=np.nan, minimum=0.0, maximum=1.0) for v in (assay_quality, lod_margin, replicate_confidence)]
    valid = [v for v in values if np.isfinite(v)]
    quality = float(np.mean(valid)) if valid else p.default_quality
    observed = 1.0 if valid else 0.0
    return {"Measurement_Quality": bounded(quality, p.minimum_quality, p.maximum_quality), "Measurement_Quality_Observed": observed}


def temporal_role(measurement_day: Any, treatment_start_day: Any = 0.0) -> str:
    day, start = safe_float(measurement_day, default=np.nan), safe_float(treatment_start_day, default=0.0)
    if not np.isfinite(day):
        return "Unknown"
    if day < start:
        return "Pre_Treatment"
    if math.isclose(day, start, abs_tol=EPS):
        return "Baseline"
    return "Post_Treatment"


def baseline_eligibility(measurement_day: Any, treatment_start_day: Any = 0.0) -> bool:
    return temporal_role(measurement_day, treatment_start_day) in {"Pre_Treatment", "Baseline"}


def pairwise_kinetic_state(tumor_fraction_base: Any, tumor_fraction_followup: Any, delta_days: Any, parameters: TCFTParameters = DEFAULT_PARAMETERS) -> Dict[str, Any]:
    f0, f1, dt = safe_float(tumor_fraction_base, default=np.nan, minimum=TINY, maximum=1.0), safe_float(tumor_fraction_followup, default=np.nan, minimum=TINY, maximum=1.0), safe_float(delta_days, default=np.nan, minimum=EPS)
    if not (np.isfinite(f0) and np.isfinite(f1) and np.isfinite(dt)):
        return {"Kinetic_Observed": 0, "Kinetic_Velocity_30d": np.nan, "Kinetic_State": 1.0}
    
    velocity = (math.log(f0) - math.log(f1)) / (dt / 30.0)
    p = parameters.kinetics
    response = stable_sigmoid(p.velocity_slope * velocity)
    state = p.velocity_lower_bound + p.velocity_range * response
    return {"Kinetic_Observed": 1, "Kinetic_Velocity_30d": float(velocity), "Kinetic_State": float(np.clip(state, 0.0, 1.0))}


def _valid_measurements(measurements: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    valid = []
    for day, fraction in measurements:
        d, f = safe_float(day, default=np.nan), safe_float(fraction, default=np.nan, minimum=TINY, maximum=1.0)
        if np.isfinite(d) and np.isfinite(f):
            valid.append((float(d), float(math.log(f))))
    valid.sort(key=lambda z: z[0])
    return valid


def _linear_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2 or np.ptp(x) <= 0:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= EPS else max(0.0, 1.0 - ss_res / ss_tot)
    return float(slope), float(r2)


def longitudinal_trajectory(measurements: Sequence[Tuple[float, float]], parameters: TCFTParameters = DEFAULT_PARAMETERS) -> Dict[str, Any]:
    valid = _valid_measurements(measurements)
    n = len(valid)

    if n < parameters.kinetics.minimum_points_for_regression:
        return {
            "Trajectory_Observed": 0, "Trajectory_N": n, "Trajectory_Velocity_30d": np.nan,
            "Trajectory_R2": np.nan, "Trajectory_Acceleration_30d": np.nan,
            "Trajectory_Rebound_Score": np.nan, "Trajectory_Plateau_Score": np.nan, "Trajectory_State": np.nan
        }

    x, y = np.asarray([v[0] for v in valid], dtype=float), np.asarray([v[1] for v in valid], dtype=float)
    slope, r2 = _linear_slope(x, y)
    velocity_30d = -slope * 30.0
    p = parameters.kinetics
    acceleration_30d = np.nan

    if n >= p.preferred_points_for_acceleration:
        midpoint = n // 2
        early_slope, _ = _linear_slope(x[:midpoint + 1], y[:midpoint + 1])
        late_slope, _ = _linear_slope(x[midpoint:], y[midpoint:])
        if np.isfinite(early_slope) and np.isfinite(late_slope):
            acceleration_30d = (-late_slope * 30.0) - (-early_slope * 30.0)

    rebound_score = 0.0
    if n >= 3:
        first_slope, _ = _linear_slope(x[:2], y[:2])
        last_slope, _ = _linear_slope(x[-2:], y[-2:])
        if np.isfinite(first_slope) and np.isfinite(last_slope) and first_slope < 0 and last_slope > 0:
            rebound_score = 1.0

    plateau_score = float(math.exp(-(abs(velocity_30d) / max(p.plateau_tolerance, EPS))))
    velocity_state = p.velocity_lower_bound + p.velocity_range * stable_sigmoid(p.velocity_slope * velocity_30d)
    
    acceleration_component = p.acceleration_weight * stable_sigmoid(p.acceleration_slope * acceleration_30d) if np.isfinite(acceleration_30d) else 0.0
    rebound_penalty = p.rebound_weight * rebound_score
    trajectory_state = float(np.clip(velocity_state + acceleration_component - rebound_penalty, 0.0, 1.0))

    return {
        "Trajectory_Observed": 1, "Trajectory_N": n, "Trajectory_Velocity_30d": float(velocity_30d),
        "Trajectory_R2": float(r2), "Trajectory_Acceleration_30d": float(acceleration_30d) if np.isfinite(acceleration_30d) else np.nan,
        "Trajectory_Rebound_Score": float(rebound_score), "Trajectory_Plateau_Score": float(plateau_score),
        "Trajectory_State": trajectory_state,
    }


def quality_coupled_kinetics(kinetic_state: float, quality_state: float, observed: bool, parameters: TCFTParameters = DEFAULT_PARAMETERS) -> float:
    if not observed:
        return 1.0
    q = bounded(quality_state, parameters.quality.minimum_quality, parameters.quality.maximum_quality)
    effective_q = max(parameters.quality.kinetic_quality_floor, q)
    return float(effective_q * kinetic_state + (1.0 - effective_q) * 1.0)


def tc_kuo_core_formula(
    *, evidence_tier: Any, evidence_confidence: Any = np.nan, therapy_match_type: Any = "Unknown",
    therapy_match_strength: Any = np.nan, kras_state: float = 0.0, tp53_state: float = 0.0, sii: Any = np.nan,
    ast: Any = np.nan, alt: Any = np.nan, age: Any = np.nan, tumor_fraction_base: Any = np.nan,
    tumor_fraction_followup: Any = np.nan, delta_days: Any = np.nan, longitudinal_measurements: Optional[Sequence[Tuple[float, float]]] = None,
    assay_quality: Any = np.nan, lod_margin: Any = np.nan, replicate_confidence: Any = np.nan,
    treatment_start_day: Any = 0.0, followup_measurement_day: Any = np.nan, parameters: TCFTParameters = DEFAULT_PARAMETERS,
) -> Dict[str, Any]:
    evidence = normalized_evidence(evidence_tier, evidence_confidence, parameters)
    compatibility = therapy_compatibility(therapy_match_type, therapy_match_strength)
    tensor = genomic_interaction_tensor(kras_state=kras_state, tp53_state=tp53_state, therapy_state=compatibility["Therapy_Compatibility"], parameters=parameters)
    genomic_utility = evidence["Evidence_Strength"] * evidence["Evidence_Confidence"] * compatibility["Therapy_Compatibility"] * tensor["Genomic_Interaction_State"]
    
    deritis, deritis_observed = calculate_deritis(ast, alt)
    host = host_state_field(sii=sii, deritis=deritis, age=age, parameters=parameters)
    quality = assay_quality_state(assay_quality=assay_quality, lod_margin=lod_margin, replicate_confidence=replicate_confidence, parameters=parameters)
    pairwise = pairwise_kinetic_state(tumor_fraction_base=tumor_fraction_base, tumor_fraction_followup=tumor_fraction_followup, delta_days=delta_days, parameters=parameters)
    
    if longitudinal_measurements is None:
        trajectory = {
            "Trajectory_Observed": 0, "Trajectory_N": 0, "Trajectory_Velocity_30d": np.nan,
            "Trajectory_R2": np.nan, "Trajectory_Acceleration_30d": np.nan,
            "Trajectory_Rebound_Score": np.nan, "Trajectory_Plateau_Score": np.nan, "Trajectory_State": np.nan,
        }
    else:
        trajectory = longitudinal_trajectory(longitudinal_measurements, parameters=parameters)

    if trajectory["Trajectory_Observed"]:
        kinetic_state, kinetic_observed, kinetic_velocity_30d = trajectory["Trajectory_State"], True, trajectory["Trajectory_Velocity_30d"]
    else:
        kinetic_state, kinetic_observed, kinetic_velocity_30d = pairwise["Kinetic_State"], bool(pairwise["Kinetic_Observed"]), pairwise["Kinetic_Velocity_30d"]

    dynamic_kinetic = quality_coupled_kinetics(kinetic_state=kinetic_state, quality_state=quality["Measurement_Quality"], observed=kinetic_observed, parameters=parameters)
    role = temporal_role(followup_measurement_day, treatment_start_day)
    post_treatment_kinetic_allowed = role == "Post_Treatment" or (not np.isfinite(safe_float(followup_measurement_day, default=np.nan)) and kinetic_observed)

    u = parameters.utility
    baseline_raw = 100.0 * (genomic_utility ** u.evidence_weight) * (host["Host_State_Field"] ** u.host_weight)
    baseline_cui = float(100.0 * np.tanh(baseline_raw / max(u.utility_scale, EPS)))

    if kinetic_observed and post_treatment_kinetic_allowed:
        dynamic_raw = baseline_raw * (dynamic_kinetic ** u.kinetic_weight)
        dynamic_status = "Dynamic_Post_Treatment"
    else:
        dynamic_raw = baseline_raw
        dynamic_status = "Baseline_No_Eligible_Kinetics"

    dynamic_cui = float(100.0 * np.tanh(dynamic_raw / max(u.utility_scale, EPS)))
    delta_cui = dynamic_cui - baseline_cui
    parameter_hash = parameter_fingerprint(parameters)
    host_complete = host["Host_Data_Completeness"]

    overall_observed = float(np.mean([
        evidence["Evidence_Observed"], compatibility["Therapy_Compatibility_Observed"],
        host_complete, quality["Measurement_Quality_Observed"], float(kinetic_observed),
    ]))

    return {
        "Model_ID": MODEL_ID, "Engine_ID": ENGINE_ID, "Architecture_Version": ARCHITECTURE_VERSION,
        "Model_Status": MODEL_STATUS, "Output_Semantics": OUTPUT_SEMANTICS, "Parameter_Fingerprint": parameter_hash,
        **evidence, **compatibility, **tensor, "Genomic_Utility": float(genomic_utility),
        "DeRitis_Ratio": float(deritis) if deritis_observed else np.nan, "DeRitis_Observed": float(deritis_observed),
        **host, **quality, **pairwise, **{f"Trajectory_{key[11:]}": value if key.startswith("Trajectory_") else value for key, value in trajectory.items()},
        "Effective_Kinetic_State": float(dynamic_kinetic), "Kinetic_Observed": int(kinetic_observed),
        "Kinetic_Velocity_30d_Used": float(kinetic_velocity_30d) if np.isfinite(kinetic_velocity_30d) else np.nan,
        "Followup_Temporal_Role": role, "Post_Treatment_Kinetic_Eligible": int(post_treatment_kinetic_allowed),
        "Baseline_Raw_Utility": float(baseline_raw), "Baseline_CUI": round(baseline_cui, 3),
        "Dynamic_Raw_Utility": float(dynamic_raw), "Dynamic_CUI": round(dynamic_cui, 3),
        "Dynamic_minus_Baseline_CUI": round(delta_cui, 3), "Utility_State": dynamic_status, "Overall_Observed_Component_Fraction": round(overall_observed, 4),
    }


def derive_feature_frame(df: pd.DataFrame, parameters: TCFTParameters = DEFAULT_PARAMETERS) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    out = df.copy()
    results: List[Dict[str, Any]] = []

    for _, row in out.iterrows():
        res = tc_kuo_core_formula(
            evidence_tier=row.get("Evidence_Tier", "Unknown"), evidence_confidence=row.get("Evidence_Confidence", np.nan),
            therapy_match_type=row.get("Therapy_Type", "Unknown"), therapy_match_strength=row.get("Therapy_Match_Strength", np.nan),
            kras_state=float(_as_bool_yes(row.get("KRAS_Mutant", "No"))), tp53_state=float(_as_bool_yes(row.get("TP53_Mutant", "No"))),
            sii=row.get("SII", np.nan), ast=row.get("AST", np.nan), alt=row.get("ALT", np.nan), age=row.get("Age", np.nan),
            tumor_fraction_base=row.get("Tumor_Fraction", np.nan), tumor_fraction_followup=row.get("Tumor_Fraction_Followup", np.nan),
            delta_days=row.get("Followup_Days", np.nan), longitudinal_measurements=None,
            assay_quality=row.get("Assay_Quality", np.nan), lod_margin=row.get("LOD_Margin", np.nan),
            replicate_confidence=row.get("Replicate_Confidence", np.nan), treatment_start_day=row.get("Treatment_Start_Day", 0.0),
            followup_measurement_day=row.get("Followup_Measurement_Day", np.nan), parameters=parameters,
        )
        results.append(res)
    return pd.concat([out, pd.DataFrame(results, index=out.index)], axis=1)


def compute_longitudinal_patient_state(measurements: Sequence[Tuple[float, float]], assay_quality: Any = np.nan, parameters: TCFTParameters = DEFAULT_PARAMETERS) -> Dict[str, Any]:
    trajectory = longitudinal_trajectory(measurements, parameters=parameters)
    quality = assay_quality_state(assay_quality=assay_quality, parameters=parameters)
    observed = bool(trajectory["Trajectory_Observed"])
    effective_state = quality_coupled_kinetics(kinetic_state=trajectory["Trajectory_State"] if observed else 1.0, quality_state=quality["Measurement_Quality"], observed=observed, parameters=parameters)
    return {**trajectory, **quality, "Effective_Longitudinal_State": float(effective_state)}


def compute_baseline_state(*, evidence_tier: Any, evidence_confidence: Any = np.nan, therapy_match_type: Any = "Unknown", therapy_match_strength: Any = np.nan, kras_state: float = 0.0, tp53_state: float = 0.0, sii: Any = np.nan, ast: Any = np.nan, alt: Any = np.nan, age: Any = np.nan, parameters: TCFTParameters = DEFAULT_PARAMETERS) -> Dict[str, Any]:
    return tc_kuo_core_formula(
        evidence_tier=evidence_tier, evidence_confidence=evidence_confidence, therapy_match_type=therapy_match_type,
        therapy_match_strength=therapy_match_strength, kras_state=kras_state, tp53_state=tp53_state,
        sii=sii, ast=ast, alt=alt, age=age, tumor_fraction_base=np.nan, tumor_fraction_followup=np.nan,
        delta_days=np.nan, longitudinal_measurements=None, assay_quality=np.nan, lod_margin=np.nan,
        replicate_confidence=np.nan, treatment_start_day=0.0, followup_measurement_day=np.nan, parameters=parameters,
    )


def model_metadata(parameters: TCFTParameters = DEFAULT_PARAMETERS) -> Dict[str, Any]:
    return {
        "Model_ID": MODEL_ID, "Engine_ID": ENGINE_ID, "Architecture_Version": ARCHITECTURE_VERSION,
        "Model_Status": MODEL_STATUS, "Output_Semantics": OUTPUT_SEMANTICS,
        "Parameter_Fingerprint": parameter_fingerprint(parameters), "Python": sys.version,
        "Platform": platform.platform(), "Numpy": np.__version__, "Pandas": pd.__version__,
        "Generated_UTC": datetime.now(timezone.utc).isoformat(),
    }


def split_temporal_roles(df: pd.DataFrame, measurement_day_column: str = "Followup_Measurement_Day", treatment_start_column: str = "Treatment_Start_Day") -> pd.DataFrame:
    out = df.copy()
    out["Temporal_Role"] = [temporal_role(row.get(measurement_day_column, np.nan), row.get(treatment_start_column, 0.0)) for _, row in out.iterrows()]
    out["Baseline_Eligible"] = [int(baseline_eligibility(row.get(measurement_day_column, np.nan), row.get(treatment_start_column, 0.0))) for _, row in out.iterrows()]
    return out


def parameter_specification(parameters: TCFTParameters = DEFAULT_PARAMETERS) -> Dict[str, Any]:
    return {"model": model_metadata(parameters), "parameters": asdict(parameters)}


def _as_bool_yes(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().casefold() in {"yes", "true", "1", "y", "present", "mutant"}


def run_self_checks() -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    checks["sigmoid_0"] = 0.0 < stable_sigmoid(0.0) < 1.0
    checks["sigmoid_large_positive"] = stable_sigmoid(1000.0) > 0.99
    checks["sigmoid_large_negative"] = stable_sigmoid(-1000.0) < 0.01
    ratio, valid = calculate_deritis(ast=25, alt=0)
    checks["invalid_alt_not_ratio_one"] = not valid and not np.isfinite(ratio)
    k = pairwise_kinetic_state(tumor_fraction_base=0.05, tumor_fraction_followup=np.nan, delta_days=30)
    checks["missing_kinetics_not_observed"] = k["Kinetic_Observed"] == 0
    k_clear = pairwise_kinetic_state(tumor_fraction_base=0.10, tumor_fraction_followup=0.05, delta_days=30)
    k_progress = pairwise_kinetic_state(tumor_fraction_base=0.05, tumor_fraction_followup=0.10, delta_days=30)
    checks["directional_kinetics"] = k_clear["Kinetic_Velocity_30d"] > k_progress["Kinetic_Velocity_30d"]
    tensor = genomic_interaction_tensor(kras_state=1, tp53_state=1, therapy_state=1)
    checks["tensor_bounded"] = 0.0 <= tensor["Genomic_Interaction_State"] <= 1.0
    baseline = compute_baseline_state(evidence_tier="Tier I", therapy_match_type="Matched", kras_state=1, tp53_state=1, sii=500, ast=25, alt=25, age=50)
    checks["baseline_mode"] = baseline["Kinetic_Observed"] == 0 and baseline["Utility_State"] == "Baseline_No_Eligible_Kinetics"
    return {"all_passed": bool(all(checks.values())), "checks": checks}
