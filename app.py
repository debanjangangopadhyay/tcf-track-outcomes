# app.py
"""TCF-001 TRACK / Precision Oncology Analytics

Revision focus:
- preserves the original Streamlit environment and eight-tab UI
- preserves the original ingestion, filtering, visualization and export features
- adds an empirically calibrated, interaction-aware clinical utility framework
- separates baseline utility from on-treatment ctDNA updating
- makes model provenance, calibration and stability visible rather than implicit
"""

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy import stats

from clinical_logic import (
    derive_feature_frame,
    compute_longitudinal_patient_state,
    model_metadata,
    parameter_specification,
    split_temporal_roles,
    run_self_checks,
    DEFAULT_PARAMETERS,
    MODEL_ID,
    ENGINE_ID,
    ARCHITECTURE_VERSION,
    MODEL_STATUS,
    OUTPUT_SEMANTICS,
)

# =========================================================
# 1. Page Configuration & Memory-Safe Styling
# =========================================================
st.set_page_config(
    page_title="TCF-001 TRACK / Precision Oncology Analytics",
    page_icon="🧬",
    layout="wide"
)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def render_download_button(fig, filename_base: str, key: str):
    """Encodes matplotlib figure directly into in-memory PDF/PNG buffers (Render-safe)."""
    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches='tight', dpi=300)
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", bbox_inches='tight', dpi=300)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="📄 Download PDF",
            data=buf_pdf.getvalue(),
            file_name=f"{filename_base}.pdf",
            mime="application/pdf",
            key=f"pdf_{key}"
        )
    with col_d2:
        st.download_button(
            label="🖼️ Download PNG",
            data=buf_png.getvalue(),
            file_name=f"{filename_base}.png",
            mime="image/png",
            key=f"png_{key}"
        )


def safe_numeric_column(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _normalise_yes_no(value):
    return "Yes" if str(value).strip().casefold() in {"yes", "y", "true", "1", "mutant"} else "No"


def _evidence_tier_from_existing(row):
    """Use only an explicitly supplied evidence tier; never infer ESCAT from KRAS alone."""
    for key in ("Evidence_Tier", "ESCAT_Tier"):
        value = row.get(key, np.nan)
        if pd.notna(value) and str(value).strip().casefold() not in {"", "nan", "none"}:
            return str(value).strip()
    return "Unknown"


def _prepare_research_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add TC-KUO v3 fields while retaining legacy application columns."""
    out = df.copy()

    if "Evidence_Tier" not in out.columns:
        out["Evidence_Tier"] = out.apply(_evidence_tier_from_existing, axis=1)
    else:
        out["Evidence_Tier"] = out["Evidence_Tier"].fillna("Unknown").astype(str).str.strip()

    if "ESCAT_Tier" not in out.columns:
        out["ESCAT_Tier"] = out["Evidence_Tier"]
    else:
        out["ESCAT_Tier"] = out["ESCAT_Tier"].fillna(out["Evidence_Tier"]).astype(str).str.strip()

    defaults = {
        "Evidence_Confidence": np.nan,
        "Therapy_Match_Strength": np.nan,
        "Assay_Quality": np.nan,
        "LOD_Margin": np.nan,
        "Replicate_Confidence": np.nan,
        "Treatment_Start_Day": 0.0,
        "Followup_Measurement_Day": np.nan,
        "Followup_Days": np.nan,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    for col in defaults:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "KRAS_Mutant" not in out.columns:
        out["KRAS_Mutant"] = "No"
    if "TP53_Mutant" not in out.columns:
        out["TP53_Mutant"] = "No"

    out["KRAS_Mutant"] = out["KRAS_Mutant"].map(_normalise_yes_no)
    out["TP53_Mutant"] = out["TP53_Mutant"].map(_normalise_yes_no)

    return out


def build_cox_features(source_df: pd.DataFrame, include_kinetics: bool = False) -> pd.DataFrame:
    """Cox design derived from the same TC-KUO v3 state variables displayed by the app."""
    d = source_df.copy()
    design = pd.DataFrame(index=d.index)

    def safe_get_series(df_obj, col_name):
        val = df_obj.get(col_name)
        if val is None:
            return pd.Series(np.nan, index=df_obj.index)
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0]
        return pd.to_numeric(val, errors="coerce")

    for col in [
        "Evidence_Strength", "Evidence_Confidence", "Therapy_Compatibility",
        "KRAS_State", "TP53_State", "Tensor_Pairwise_Order",
        "Tensor_Third_Order", "Host_State_Field", "Measurement_Quality"
    ]:
        design[col] = safe_get_series(d, col)

    def z(col):
        x = safe_get_series(d, col)
        sd = x.std(ddof=0)
        return (x - x.mean()) / (sd if np.isfinite(sd) and sd > 0 else 1.0)

    design["SII_z"] = z("SII")
    design["DeRitis_z"] = z("DeRitis_Ratio")
    design["Age_z"] = z("Age")

    if include_kinetics:
        for col in ["ctDNA_Log_Ratio", "Effective_Kinetic_State", "Kinetic_Observed"]:
            design[col] = safe_get_series(d, col)

    return design.replace([np.inf, -np.inf], np.nan)
    

def fit_cox_design(source_df: pd.DataFrame, include_kinetics: bool = False):
    """Fit a stabilized research Cox model while dropping zero-variance predictors."""
    design = build_cox_features(source_df, include_kinetics=include_kinetics)
    outcome = pd.DataFrame({
        "PFS_Months": pd.to_numeric(source_df["PFS_Months"], errors="coerce"),
        "Progression_Event": pd.to_numeric(source_df["Progression_Event"], errors="coerce"),
    })
    model_df = pd.concat([outcome, design], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

    feature_cols = [
        c for c in design.columns
        if c in model_df.columns and model_df[c].nunique(dropna=True) > 1
    ]
    if not feature_cols or model_df["Progression_Event"].sum() < 2:
        return None, model_df, feature_cols

    model_df = model_df[["PFS_Months", "Progression_Event"] + feature_cols].copy()

    try:
        cph = CoxPHFitter(penalizer=0.08, l1_ratio=0.0)
        cph.fit(model_df, duration_col="PFS_Months", event_col="Progression_Event")
        return cph, model_df, feature_cols
    except Exception:
        try:
            cph = CoxPHFitter()
            cph.fit(model_df, duration_col="PFS_Months", event_col="Progression_Event")
            return cph, model_df, feature_cols
        except Exception:
            return None, model_df, feature_cols


def bootstrap_c_index(source_df: pd.DataFrame, include_kinetics: bool = False, n_boot: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(source_df)
    if n < 15:
        return np.nan, np.nan, np.nan

    values = []
    for _ in range(n_boot):
        boot = source_df.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
        cph, model_df, cols = fit_cox_design(boot, include_kinetics=include_kinetics)
        if cph is not None and cols:
            value = getattr(cph, "concordance_index_", np.nan)
            if np.isfinite(value):
                values.append(float(value))

    if len(values) < 5:
        return np.nan, np.nan, np.nan
    arr = np.asarray(values)
    return float(np.median(arr)), float(np.quantile(arr, .025)), float(np.quantile(arr, .975))


def compute_empirical_cui(source_df: pd.DataFrame, coefficients=None, baseline: float = 50.0):
    """Optional empirical calibration layered over the mechanistic TC-KUO v3 index."""
    out = source_df.copy()
    mechanistic = pd.to_numeric(out["Dynamic_CUI"], errors="coerce").fillna(0.0)
    out["Mechanistic_CUI"] = mechanistic

    if coefficients is None:
        out["Calibrated_CUI"] = mechanistic
        out["CUI_Calibration"] = "Mechanistic TC-KUO v3 research index"
        return out

    design = build_cox_features(out, include_kinetics=False).reindex(columns=coefficients.index, fill_value=0.0)
    linear = design.astype(float).fillna(0.0).dot(coefficients.astype(float))
    calibrated = 100.0 / (1.0 + np.exp(-np.clip(linear, -20, 20)))
    out["Calibrated_CUI"] = calibrated.astype(float)
    out["CUI_Calibration"] = "Cohort-derived Cox calibration over TC-KUO v3 state features"
    out["Calibration_Delta"] = out["Calibrated_CUI"] - mechanistic
    return out


def score_explanation(row):
    return {
        "Actionability": float(row.get("Evidence_Strength", 0.0)) * 100.0,
        "Therapy Match": float(row.get("Therapy_Compatibility", 0.0)),
        "Resistance Modifier": float(row.get("Tensor_Pairwise_Order", 0.0)),
        "Host Resilience": float(row.get("Host_State_Field", 0.0)),
        "Toxicity Modifier": float(row.get("Toxicity_Modifier", 1.0)),
    }


# =========================================================
# 2. Data Ingestion & Flexible Schema Mapping
# =========================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3022/3022565.png", width=60)
st.sidebar.title("TCF-001 TRACK")
st.sidebar.markdown("---")

data_mode = st.sidebar.radio("Data Source:", ["📂 Upload Custom Cohort", "🔬 Load Validation Demo"])


@st.cache_data
def process_dataframe(df):
    """Normalize the cohort and execute the complete TC-KUO v3 research engine."""
    df = _prepare_research_schema(df)

    df["PFS_Months"] = safe_numeric_column(df, "PFS_Months", np.nan)
    df["Progression_Event"] = safe_numeric_column(df, "Progression_Event", np.nan)
    df["Tumor_Fraction"] = safe_numeric_column(df, "Tumor_Fraction", np.nan)
    df = df.dropna(subset=["PFS_Months", "Progression_Event"]).copy()

    if "Cohort" not in df.columns:
        df["Cohort"] = "General Cohort"
    if "Therapy_Type" not in df.columns:
        df["Therapy_Type"] = "Standard Care"
    if "SII" not in df.columns:
        df["SII"] = np.nan
    if "AST" not in df.columns:
        df["AST"] = np.nan
    if "ALT" not in df.columns:
        df["ALT"] = np.nan
    if "Age" not in df.columns:
        df["Age"] = np.nan
    if "Tumor_Fraction_Followup" not in df.columns:
        df["Tumor_Fraction_Followup"] = np.nan

    for col in ["KRAS_Mutant", "TP53_Mutant", "Therapy_Type", "Cohort"]:
        df[col] = df[col].astype(str).str.strip()

    for col in [
        "SII", "AST", "ALT", "Age", "Tumor_Fraction",
        "Tumor_Fraction_Followup", "Followup_Days",
        "Treatment_Start_Day", "Followup_Measurement_Day"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Explicit evidence is preserved. No automatic KRAS-only ESCAT inference is performed.
    df["Evidence_Tier"] = df["Evidence_Tier"].fillna("Unknown")
    df["ESCAT_Tier"] = df["Evidence_Tier"]

    df = split_temporal_roles(df)
    df = derive_feature_frame(df)

    # Backward-compatible public columns retained for the original application.
    df["VMTB_Matching_Score"] = df["Dynamic_CUI"].round(1)
    df["Actionability_Points"] = pd.to_numeric(df["Evidence_Strength"], errors="coerce").fillna(0) * 100
    df["Therapy_Match"] = df["Therapy_Compatibility"]
    df["TP53_Resistance_Base"] = 1.0 - 0.25 * pd.to_numeric(df["TP53_State"], errors="coerce").fillna(0)
    df["Phi_Host"] = df["Host_State_Field"]
    return df


raw_df = None
benchmark_path = os.path.join("data", "Processed_Clinical_Dashboard_Data.xlsx")

if data_mode == "📂 Upload Custom Cohort":
    uploaded_file = st.sidebar.file_uploader("Upload Clinical File (.xlsx, .csv)", type=['xlsx', 'xls', 'csv'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            raw_upload = pd.read_csv(io.BytesIO(uploaded_file.read()))
        else:
            raw_upload = pd.read_excel(io.BytesIO(uploaded_file.read()))

        st.sidebar.markdown("### 🔄 Map Dataset Columns")
        cols = ["Not Available"] + list(raw_upload.columns)

        def get_idx(col_name):
            return cols.index(col_name) if col_name in cols else 0

        map_pfs = st.sidebar.selectbox("PFS (Months)", cols, index=get_idx('PFS_Months'))
        map_evt = st.sidebar.selectbox("Progression Event", cols, index=get_idx('Progression_Event'))
        map_tx = st.sidebar.selectbox("Therapy Administered", cols, index=get_idx('Therapy_Type'))
        map_coh = st.sidebar.selectbox("Cancer Cohort", cols, index=get_idx('Cohort'))
        map_kras = st.sidebar.selectbox("KRAS Status", cols, index=get_idx('KRAS_Mutant'))
        map_tp53 = st.sidebar.selectbox("TP53 Status", cols, index=get_idx('TP53_Mutant'))

        st.sidebar.markdown("#### Host-Tumor Modifiers")
        map_ast = st.sidebar.selectbox("AST Level (MASLD Proxy)", cols, index=get_idx('AST'))
        map_age = st.sidebar.selectbox("Patient Age (Toxicity)", cols, index=get_idx('Age'))
        map_followup = st.sidebar.selectbox("Follow-up ctDNA % (Kinetics)", cols, index=get_idx('Tumor_Fraction_Followup'))

        if st.sidebar.button("Process & Analyze Data"):
            if map_pfs == "Not Available" or map_evt == "Not Available":
                st.sidebar.error("❌ Critical: You MUST map 'PFS (Months)' and 'Progression Event' to run survival analytics.")
            else:
                rename_dict = {}
                for map_val, target in zip(
                    [map_pfs, map_evt, map_tx, map_coh, map_kras, map_tp53, map_ast, map_age, map_followup],
                    ['PFS_Months', 'Progression_Event', 'Therapy_Type', 'Cohort', 'KRAS_Mutant', 'TP53_Mutant', 'AST', 'Age', 'Tumor_Fraction_Followup']
                ):
                    if map_val != "Not Available":
                        rename_dict[map_val] = target

                mapped_df = raw_upload.rename(columns=rename_dict)
                raw_df = process_dataframe(mapped_df)
                st.session_state['mapped_df'] = raw_df
        elif 'mapped_df' in st.session_state:
            raw_df = st.session_state['mapped_df']
        else:
            st.info("👈 Please map your columns and click 'Process & Analyze Data'.")
            st.stop()
    else:
        st.info("👈 Please upload a dataset in the sidebar to begin.")
        st.stop()
else:
    if os.path.exists(benchmark_path):
        raw_df = process_dataframe(pd.read_excel(benchmark_path))
        st.sidebar.success("Validation Cohort Loaded.")
    else:
        st.error("Validation file not found in repository.")
        st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Clinical Filters")
cohort_options = sorted(raw_df['Cohort'].dropna().unique().tolist())
selected_cohorts = st.sidebar.multiselect("Filter Cohort", options=cohort_options, default=cohort_options)
kras_filter = st.sidebar.selectbox("KRAS Status", ['All', 'Yes', 'No'])

filtered_df = raw_df[raw_df['Cohort'].isin(selected_cohorts)].copy()
if kras_filter != 'All':
    filtered_df = filtered_df[filtered_df['KRAS_Mutant'] == kras_filter].copy()

total_patients = len(filtered_df)
if total_patients == 0:
    st.warning("No patients match current filter parameters.")
    st.stop()

# =========================================================
# 3. Empirical calibration layer
# =========================================================
baseline_cph, baseline_model_df, baseline_cols = fit_cox_design(filtered_df, include_kinetics=False)
if baseline_cph is not None:
    calibrated_df = compute_empirical_cui(filtered_df, baseline_cph.params_, baseline=50.0)
else:
    calibrated_df = compute_empirical_cui(filtered_df, None)

# A dynamic model is only fitted when follow-up kinetics are sufficiently observed.
kinetic_observed = calibrated_df['K_Observed'].fillna(0).astype(float).sum()
if kinetic_observed >= 8:
    dynamic_cph, dynamic_model_df, dynamic_cols = fit_cox_design(calibrated_df.dropna(subset=['ctDNA_Log_Ratio']), include_kinetics=True)
else:
    dynamic_cph, dynamic_model_df, dynamic_cols = None, pd.DataFrame(), []

if baseline_cph is not None:
    calibrated_df = compute_empirical_cui(calibrated_df, baseline_cph.params_)
else:
    calibrated_df['Calibrated_CUI'] = calibrated_df['Dynamic_CUI'].astype(float)
    calibrated_df['CUI_Calibration'] = 'Mechanistic fallback (insufficient stable Cox information)'

# Keep the filtered frame name used throughout the original UI.
filtered_df = calibrated_df.copy()

# =========================================================
# 4. Header Metrics
# =========================================================
st.title("Decentralized Genomic Profiling & Clinical Analytics")
st.caption("Integrative Host-Tumor Platform: ESCAT, Epistasis, Toxicity (Tau), & Clonal Kinetics")

col1, col2, col3, col4 = st.columns(4)
matched_pts = filtered_df[filtered_df['Therapy_Type'] == 'Targeted/Matched']
matched_rate = round((len(matched_pts) / total_patients) * 100, 1) if total_patients else 0.0

col1.metric("Evaluable Cohort", f"{total_patients:,} pts")
col2.metric("Median PFS", f"{filtered_df['PFS_Months'].median():.1f} Mo")
col3.metric("Targeted Therapy Rate", f"{matched_rate}%")
col4.metric("Mean Clinical Utility (CUI)", f"{filtered_df['Calibrated_CUI'].mean():.1f}")
st.markdown("---")
st.info(
    f"🧪 **Research Use Only — {MODEL_ID} / {ENGINE_ID} v{ARCHITECTURE_VERSION}.** "
    "CUI is a bounded research utility index, not a probability, diagnosis, prognosis, "
    "treatment recommendation, or clinical decision rule. Evidence tiers are externally curated."
)

# =========================================================
# 5. Tabbed Analytical Interface — preserved exactly
# =========================================================
tab_surv, tab_cph, tab_nomogram, tab_vmtb, tab_mrd, tab_mut, tab_pathway, tab_data = st.tabs([
    "📈 Survival & Benchmarks",
    "🌲 Multivariable Cox PH",
    "🧮 Interactive Nomogram",
    "🎯 ESCAT & CUI",
    "🔬 Liquid Biopsy Kinetics",
    "🧬 Mutation Co-Occurrence",
    "🗺️ Decision Pathway",
    "📋 Data Export"
])

# =========================================================
# TAB 1: Kaplan–Meier & Benchmarks
# =========================================================
with tab_surv:
    st.subheader("Progression-Free Survival & Global Benchmark Overlay")
    matched = filtered_df[filtered_df['Therapy_Type'] == 'Targeted/Matched']
    unmatched = filtered_df[filtered_df['Therapy_Type'] == 'Standard Care']

    if len(matched) > 0 and len(unmatched) > 0:
        fig_km, ax_km = plt.subplots(figsize=(8, 4.8))
        kmf = KaplanMeierFitter()

        kmf.fit(matched['PFS_Months'], matched['Progression_Event'], label=f'Targeted (n={len(matched)})')
        kmf.plot_survival_function(ax=ax_km, color='#1f77b4', ci_show=True, lw=2.5)

        kmf.fit(unmatched['PFS_Months'], unmatched['Progression_Event'], label=f'Standard Care (n={len(unmatched)})')
        kmf.plot_survival_function(ax=ax_km, color='#d62728', ci_show=True, lw=2.5)

        if os.path.exists(benchmark_path) and data_mode == "📂 Upload Custom Cohort":
            bench_df = pd.read_excel(benchmark_path)
            bench_df['PFS_Months'] = pd.to_numeric(bench_df.get('PFS_Months'), errors='coerce')
            bench_df['Progression_Event'] = pd.to_numeric(bench_df.get('Progression_Event'), errors='coerce')
            bench_df = bench_df.dropna(subset=['PFS_Months', 'Progression_Event'])
            if len(bench_df) > 0:
                kmf.fit(bench_df['PFS_Months'], bench_df['Progression_Event'], label=f'TCGA Baseline (n={len(bench_df)})')
                kmf.plot_survival_function(ax=ax_km, color='gray', linestyle='--', alpha=0.6)

        ax_km.set_title("Survival Probability vs. Baselines", fontsize=12)
        ax_km.set_xlabel("Progression-Free Interval (Months)")
        ax_km.set_ylabel("Probability of PFS $S(t)$")
        ax_km.grid(axis='y', linestyle='--', alpha=0.5)

        st.pyplot(fig_km)
        render_download_button(fig_km, "KM_Benchmark_Overlay", key="km_bench")
        plt.close(fig_km)

        try:
            lr = logrank_test(
                matched['PFS_Months'], unmatched['PFS_Months'],
                event_observed_A=matched['Progression_Event'],
                event_observed_B=unmatched['Progression_Event']
            )
            st.caption(f"Log-rank comparison: statistic={lr.test_statistic:.3f}, p={lr.p_value:.4g}")
        except Exception:
            pass
    else:
        st.info("Insufficient variance to plot survival curves.")

# =========================================================
# TAB 2: Multivariable Cox PH — now includes interactions + host state
# =========================================================
with tab_cph:
    st.subheader("Multivariable Cox Proportional Hazards Regression")
    st.markdown(
        "Baseline model includes therapeutic matching, KRAS/TP53 state, explicit co-mutation interactions, "
        "and continuous host-state modifiers. This replaces the earlier binary-only covariate treatment."
    )

    if baseline_cph is None or len(baseline_cols) == 0:
        st.info("⚠️ Insufficient event diversity or predictor variance to fit a stable multivariable Cox model.")
    else:
        col_cph1, col_cph2 = st.columns([1.2, 1])
        with col_cph1:
            fig_cph, ax_cph = plt.subplots(figsize=(7, 4.8))
            baseline_cph.plot(ax=ax_cph)
            ax_cph.set_title("Adjusted Hazard Ratios (95% CI)", fontsize=12)
            ax_cph.grid(axis='x', linestyle='--', alpha=0.5)
            st.pyplot(fig_cph)
            render_download_button(fig_cph, "Cox_PH_Forest_Plot", key="cph")
            plt.close(fig_cph)

        with col_cph2:
            summary_table = baseline_cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].reset_index()
            summary_table.columns = ['Covariate', 'Log-Hazard', 'Hazard Ratio', 'Std Error', 'p-value']
            st.dataframe(
                summary_table.style.format({
                    'Log-Hazard': '{:.3f}',
                    'Hazard Ratio': '{:.3f}',
                    'Std Error': '{:.3f}',
                    'p-value': '{:.4e}'
                }),
                use_container_width=True
            )

        cidx = getattr(baseline_cph, 'concordance_index_', np.nan)
        b1, b2, b3 = st.columns(3)
        b1.metric("Apparent C-index", f"{cidx:.3f}" if np.isfinite(cidx) else "NA")
        b2.metric("Events", f"{int(filtered_df['Progression_Event'].sum())}")
        b3.metric("Predictors", f"{len(baseline_cols)}")

        med_c, lo_c, hi_c = bootstrap_c_index(filtered_df, include_kinetics=False, n_boot=60)
        if np.isfinite(med_c):
            st.caption(f"Bootstrap stability check: median C-index {med_c:.3f} (95% empirical interval {lo_c:.3f}–{hi_c:.3f}).")
        st.info("Model interpretation: the interaction terms are statistical effect-modifiers; this engine does not label them as biological epistasis without external mechanistic evidence.")

# =========================================================
# TAB 3: Interactive Nomogram — host/tumor variables are real predictors
# =========================================================
with tab_nomogram:
    st.subheader("Interactive Research Nomogram")
    st.markdown("Translates the cohort's fitted statistical model into an exploratory research curve; it is not a point-of-care or treatment-decision tool.")

    col_n1, col_n2 = st.columns([1, 2])
    with col_n1:
        st.write("**Patient Parameters**")
        pt_therapy = st.radio("Therapy Administered", ["Targeted/Matched", "Standard Care"])
        pt_kras = st.selectbox("Actionable Target (KRAS)", ["Yes", "No"])
        pt_tp53 = st.selectbox("Resistance Co-Mutation (TP53)", ["No", "Yes"])
        pt_age = st.slider("Patient Age (Toxicity Modifier)", 30, 90, 50)
        pt_sii = st.slider("Systemic Inflammation (SII)", 200, 1500, 500)
        pt_deritis = st.slider("MASLD Proxy (AST/ALT Ratio)", 0.5, 3.0, 1.0, 0.1)

        patient_row = pd.Series({
            'Therapy_Type': pt_therapy,
            'KRAS_Mutant': pt_kras,
            'TP53_Mutant': pt_tp53,
            'Age': pt_age,
            'SII': pt_sii,
            'AST': pt_deritis * 25.0,
            'ALT': 25.0,
            'Cohort': 'General Cohort',
            'Tumor_Fraction': 0.05,
            'Tumor_Fraction_Followup': np.nan,
        })
        patient_features = derive_feature_frame(pd.DataFrame([patient_row]))

        pt_cui = float(patient_features['Baseline_CUI'].iloc[0])
        st.metric("Baseline Mechanistic CUI", f"{pt_cui:.1f}")
        expn = score_explanation(patient_features.iloc[0])
        st.caption(
            f"Actionability={expn['Actionability']:.0f}; Match={expn['Therapy Match']:.2f}; "
            f"Resistance={expn['Resistance Modifier']:.2f}; Host={expn['Host Resilience']:.2f}; "
            f"Toxicity={expn['Toxicity Modifier']:.2f}."
        )

    with col_n2:
        try:
            if baseline_cph is None or len(baseline_cols) == 0:
                st.info("⚠️ Insufficient cohort variance to construct a cohort-derived predictive nomogram for this filter.")
            else:
                tmp = pd.DataFrame([{
                    'Therapy_Type': pt_therapy,
                    'KRAS_Mutant': pt_kras,
                    'TP53_Mutant': pt_tp53,
                    'SII': pt_sii,
                    'DeRitis': pt_deritis,
                    'Age': pt_age,
                    'ctDNA_Log_Ratio': 0.0,
                    'K_Clearance': 1.0,
                }])
                tmp['Is_Matched'] = (tmp['Therapy_Type'] == 'Targeted/Matched').astype(float)
                tmp['KRAS_Mut'] = (tmp['KRAS_Mutant'] == 'Yes').astype(float)
                tmp['TP53_Mut'] = (tmp['TP53_Mutant'] == 'Yes').astype(float)
                tmp['KRAS_x_TP53'] = tmp['KRAS_Mut'] * tmp['TP53_Mut']
                tmp['Matched_x_TP53'] = tmp['Is_Matched'] * tmp['TP53_Mut']
                tmp['SII_z'] = (tmp['SII'] - filtered_df['SII'].mean()) / (filtered_df['SII'].std(ddof=0) if filtered_df['SII'].std(ddof=0) else 1.0)
                dr_std = filtered_df['DeRitis'].std(ddof=0) if filtered_df['DeRitis'].std(ddof=0) else 1.0
                tmp['DeRitis_z'] = (tmp['DeRitis'] - filtered_df['DeRitis'].mean()) / dr_std
                age_std = filtered_df['Age'].std(ddof=0) if filtered_df['Age'].std(ddof=0) else 1.0
                tmp['Age_z'] = (tmp['Age'] - filtered_df['Age'].mean()) / age_std
                for c in baseline_cols:
                    if c not in tmp.columns:
                        tmp[c] = 0.0
                pt_data = tmp[baseline_cols].astype(float)
                pt_survival = baseline_cph.predict_survival_function(pt_data)

                fig_nomo, ax_nomo = plt.subplots(figsize=(7, 4))
                ax_nomo.plot(pt_survival.index, pt_survival.iloc[:, 0], color='darkgreen', linewidth=2.5)
                ax_nomo.set_title("Predicted Individual Survival Trajectory $S(t | Z)$", fontsize=12)
                ax_nomo.set_xlabel("Progression-Free Interval (Months)")
                ax_nomo.set_ylabel("Probability")
                ax_nomo.grid(True, linestyle='--', alpha=0.5)

                if dynamic_cph is not None:
                    st.caption("Dynamic ctDNA updating is available for cohorts with sufficient observed follow-up kinetics and is intentionally kept distinct from the baseline curve.")
                st.pyplot(fig_nomo)
                render_download_button(fig_nomo, "Patient_Survival_Nomogram", key="nomo")
                plt.close(fig_nomo)
        except Exception as e:
            st.warning(f"Insufficient cohort variance to fit the predictive nomogram. Error: {e}")

# =========================================================
# TAB 4: ESCAT & CUI — preserves original views, adds decomposition/calibration
# =========================================================
with tab_vmtb:
    st.subheader("Integrative Actionability Distribution (CUI)")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_match = px.histogram(
            filtered_df, x='Calibrated_CUI', color='ESCAT_Tier',
            nbins=20, barmode='stack', title="Clinical Utility Index (CUI) by ESCAT Tier",
            category_orders={"ESCAT_Tier": ["Tier I", "Tier II", "Tier III", "Tier IV"]}
        )
        fig_match.add_vline(x=50, line_dash="dash", line_color="green")
        st.plotly_chart(fig_match, use_container_width=True)
    with col_v2:
        display_df = filtered_df.copy()
        display_df['Match_Tier'] = np.where(display_df['Calibrated_CUI'] >= 50, 'High Utility (≥50)', 'Low Utility (<50)')
        fig_box = px.box(
            display_df, x='Match_Tier', y='PFS_Months', color='Match_Tier',
            color_discrete_sequence=['#2ca02c', '#7f7f7f'], title="PFS by Utility Threshold"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### CUI Component Attribution")
    attribution_cols = [
        'Evidence_Strength', 'Therapy_Compatibility', 'Tensor_Pairwise_Order',
        'Host_State_Field', 'Measurement_Quality', 'Effective_Kinetic_State', 'Baseline_CUI',
        'Dynamic_CUI', 'Calibrated_CUI'
    ]
    st.dataframe(filtered_df[attribution_cols].describe().T.round(3), use_container_width=True)
    st.caption("The calibrated CUI combines the transparent mechanistic state with stable cohort-derived Cox effects when available; otherwise it falls back to the mechanistic score.")

# =========================================================
# TAB 5: Liquid Biopsy — original view + kinetic classification
# =========================================================
with tab_mrd:
    st.subheader("superRCA Liquid Biopsy: Circulating Tumor Fraction vs PFS")
    fig_mrd = go.Figure()
    fig_mrd.add_trace(go.Scatter(
        x=filtered_df['PFS_Months'], y=filtered_df['Tumor_Fraction'], mode='markers',
        marker=dict(size=8, color=filtered_df['Calibrated_CUI'], colorscale='Viridis', showscale=True, colorbar=dict(title="CUI Score")),
        text=filtered_df.get('Patient_ID', pd.Series(['ID'] * len(filtered_df), index=filtered_df.index)),
        hovertemplate="<b>Patient:</b> %{text}<br><b>PFS:</b> %{x:.1f} Mo<br><b>Tumor Fraction:</b> %{y:.4f}%<extra></extra>"
    ))
    fig_mrd.update_layout(xaxis_title='Progression-Free Survival (Months)', yaxis_title='Tumor Fraction % (Log Scale)', yaxis_type="log")
    fig_mrd.add_hline(y=0.01, line_dash="dash", line_color="red", annotation_text="superRCA LOD (0.01%)")
    st.plotly_chart(fig_mrd, use_container_width=True)

    st.markdown("#### Multi-timepoint trajectory analysis")
    st.caption(
        "Optional research analysis. Enter ordered measurements as day:fraction, comma-separated "
        "(example: 0:0.08,30:0.04,60:0.06). Trajectory descriptors are not clinical resistance labels."
    )
    trajectory_text = st.text_input("Patient trajectory (optional)", value="", key="trajectory_input")
    if trajectory_text.strip():
        try:
            pairs = []
            for token in trajectory_text.split(","):
                day, frac = token.strip().split(":")
                pairs.append((float(day), float(frac)))
            traj = compute_longitudinal_patient_state(pairs)
            st.dataframe(pd.DataFrame([traj]).T.rename(columns={0: "Value"}), use_container_width=True)
        except Exception as exc:
            st.warning(f"Trajectory could not be parsed: {exc}")

    observed_kin = filtered_df[filtered_df['Kinetic_Observed'] > 0].copy()
    if len(observed_kin) > 0:
        st.markdown("#### Longitudinal Molecular Update")
        st.dataframe(
            observed_kin[[
                'Tumor_Fraction', 'Tumor_Fraction_Followup',
                'Effective_Kinetic_State', 'Dynamic_CUI'
            ]].round(4),
            use_container_width=True
        )
        if dynamic_cph is not None:
            st.info("Dynamic Cox layer enabled: the ctDNA log-ratio is estimated as a time-varying molecular-response signal for cohorts with adequate follow-up observations. The present interface remains a proof-of-concept and does not establish prospective causal treatment adaptation.")
    else:
        st.info("No adequate baseline-to-follow-up ctDNA pairs were detected; baseline CUI remains active and no longitudinal update is inferred.")

# =========================================================
# TAB 6: Mutation Co-occurrence — preserved + interaction language cleaned up
# =========================================================
with tab_mut:
    st.subheader("Variant Co-occurrence (Fisher's Exact Test)")
    if 'TP53_Mutant' in filtered_df.columns and 'KRAS_Mutant' in filtered_df.columns:
        contingency = pd.crosstab(filtered_df['TP53_Mutant'], filtered_df['KRAS_Mutant'])
        if contingency.shape == (2, 2):
            odds_ratio, p_val = stats.fisher_exact(contingency)
            col_m1, col_m2 = st.columns([1, 1.2])
            with col_m1:
                st.write(contingency)
                st.metric("Odds Ratio (OR)", f"{odds_ratio:.2f}")
                st.metric("Fisher's Exact p-value", f"{p_val:.4e}")
            with col_m2:
                fig_heat, ax_heat = plt.subplots(figsize=(5, 3))
                sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_heat)
                ax_heat.set_title("TP53 vs KRAS Co-Occurrence", fontsize=10)
                st.pyplot(fig_heat)
                plt.close(fig_heat)

            if baseline_cph is not None and 'Tensor_Pairwise_Order' in baseline_cph.params_.index:
                interaction_coef = baseline_cph.params_['Tensor_Pairwise_Order']
                interaction_hr = np.exp(interaction_coef)
                st.metric("Estimated Interaction HR", f"{interaction_hr:.2f}")
                st.caption("This is an empirical statistical interaction term; it is not treated as mechanistic epistasis by the software unless independently established.")
        else:
            st.info("Insufficient variance to compute 2x2 matrix.")
    else:
        st.info("Missing `TP53_Mutant` or `KRAS_Mutant` columns.")

# =========================================================
# TAB 7: Decision Pathway — original structure, refined semantics
# =========================================================
with tab_pathway:
    st.subheader("Dynamic Research State & Actionability Pathway")
    st.markdown("Research-state flowchart mapping curated evidence, therapy compatibility, genomic interaction, host state, assay quality, and longitudinal molecular kinetics.")

    tier_counts = filtered_df['ESCAT_Tier'].value_counts()
    t12_count = tier_counts.get('Tier I', 0) + tier_counts.get('Tier II', 0)
    t12_pct = round((t12_count / total_patients) * 100, 1) if total_patients else 0
    t34_pct = round(100 - t12_pct, 1)

    tp53_count = len(filtered_df[filtered_df['TP53_Mutant'] == 'Yes'])
    epistasis_pct = round((tp53_count / total_patients) * 100, 1) if total_patients else 0

    kinetic_pct = round((kinetic_observed / total_patients) * 100, 1) if total_patients else 0
    mean_cui = float(filtered_df['Calibrated_CUI'].mean())

    dot_graph = f"""
    digraph ClinicalPathway {{
        rankdir=TB;
        node [shape=box, style=\"filled,rounded\", fontname=\"Helvetica\", fontsize=10];
        edge [fontname=\"Helvetica\", fontsize=9, color=\"#555555\"];

        A [label=\"Filtered Cohort Evaluated\\nn = {total_patients}\", fillcolor=\"#cce5ff\"];
        B [label=\"ESCAT-Inspired Mutational Mapping\\n(Target & Cohort Correlation)\", fillcolor=\"#e2e3e5\"];

        C1 [label=\"Tier I / II\\nActionability\", fillcolor=\"#d4edda\", color=\"#28a745\"];
        C2 [label=\"Tier III / IV\\nLower/Investigational\", fillcolor=\"#f8d7da\", color=\"#dc3545\"];

        D [label=\"KRAS×TP53 Interaction Context\\nTP53 Co-mutation: {epistasis_pct}%\", fillcolor=\"#fff3cd\"];
        E [label=\"Host-Tumor State + Toxicity\\nDe Ritis, SII, Age\", fillcolor=\"#e0c3fc\"];

        K [label=\"Longitudinal Molecular Update\\nctDNA observed: {kinetic_pct}%\", fillcolor=\"#ffe8a1\"];
        F [label=\"Final Clinical Utility Index (CUI)\\nMean calibrated CUI: {mean_cui:.1f}\", fillcolor=\"#d1e7dd\"];

        A -> B;
        B -> C1 [label=\"{t12_pct}%\"];
        B -> C2 [label=\"{t34_pct}%\"];

        C1 -> D [label=\"Therapeutic Matching\"];
        C2 -> D;
        D -> E [label=\"Resistance interaction\"];
        E -> K [label=\"Host resilience & toxicity\"];
        K -> F [label=\"Dynamic molecular update\"];
    }}
    """
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        st.graphviz_chart(dot_graph, use_container_width=True)
    with col_g2:
        st.info("**Flowchart Export**\n\nThis graph dynamically adapts to your uploaded cohort. You can right-click or drag the flowchart to save it directly as an SVG/PNG for manuscript figure integration.")

# =========================================================
# TAB 8: Data Export — preserve original dataset + add model provenance
# =========================================================
with tab_data:
    st.subheader("Cohort Dataset & Table 1 Summary")
    st.dataframe(filtered_df, use_container_width=True)

    tier_counts = filtered_df['ESCAT_Tier'].value_counts()

    table_1_data = {
        "Metric": [
            "Total Patients",
            "Median PFS (Months, IQR)",
            "Progressive Events (%)",
            "Targeted Therapy (%)",
            "Mean CUI Score",
            "Tier I / Tier II Actionability (%)"
        ],
        "Value": [
            f"{total_patients}",
            f"{filtered_df['PFS_Months'].median():.1f} ({filtered_df['PFS_Months'].quantile(0.25):.1f} - {filtered_df['PFS_Months'].quantile(0.75):.1f})",
            f"{(filtered_df['Progression_Event'].sum() / total_patients)*100:.1f}%",
            f"{matched_rate}%",
            f"{filtered_df['Calibrated_CUI'].mean():.1f}",
            f"{((tier_counts.get('Tier I', 0) + tier_counts.get('Tier II', 0)) / total_patients * 100):.1f}%"
        ]
    }
    table_1_df = pd.DataFrame(table_1_data)
    st.table(table_1_df)

    st.markdown("#### TC-KUO v3 Research Architecture & Provenance")
    st.json(model_metadata(DEFAULT_PARAMETERS))
    st.caption(
        "The export retains parameter fingerprinting, temporal provenance, observed-component fraction, "
        "measurement quality, and separate baseline/dynamic utility fields."
    )

    st.markdown("#### Model Provenance")
    provenance = pd.DataFrame({
        'Element': [
            'Actionability mapping', 'Resistance feature', 'Host state',
            'Toxicity', 'ctDNA kinetic update', 'Calibration layer'
        ],
        'Implementation': [
            'Externally curated evidence tier; no KRAS-only inference',
            'Sparse first-/second-/third-order genomic interaction tensor',
            'Continuous host-state field with explicit missingness',
            'Host-state + assay-quality measurement modifiers',
            'Pairwise + longitudinal ctDNA kinetics with temporal provenance',
            'Optional penalized Cox calibration over v3 state variables'
        ]
    })
    st.dataframe(provenance, use_container_width=True)

    csv_buffer = io.BytesIO()
    table_1_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Table 1 (CSV)", data=csv_buffer.getvalue(), file_name="Table1_Baseline_Characteristics.csv", mime="text/csv")
