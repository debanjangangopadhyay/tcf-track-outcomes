"""TCF-001 TRACK / Precision Oncology Analytics"""

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

st.set_page_config(page_title="TCF-001 TRACK / Precision Oncology Analytics", page_icon="🧬", layout="wide")

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def render_download_button(fig, filename_base: str, key: str):
    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches='tight', dpi=300)
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", bbox_inches='tight', dpi=300)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(label="📄 Download PDF", data=buf_pdf.getvalue(), file_name=f"{filename_base}.pdf", mime="application/pdf", key=f"pdf_{key}")
    with col_d2:
        st.download_button(label="🖼️ Download PNG", data=buf_png.getvalue(), file_name=f"{filename_base}.png", mime="image/png", key=f"png_{key}")


def safe_numeric_column(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    val = df[col]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return pd.to_numeric(val, errors="coerce").fillna(default)


def safe_get_series(df_obj, col_name):
    val = df_obj.get(col_name)
    if val is None:
        return pd.Series(np.nan, index=df_obj.index)
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return pd.to_numeric(val, errors="coerce")


def _normalise_yes_no(value):
    return "Yes" if str(value).strip().casefold() in {"yes", "y", "true", "1", "mutant"} else "No"


def _evidence_tier_from_existing(row):
    for key in ("Evidence_Tier", "ESCAT_Tier"):
        value = row.get(key, np.nan)
        if pd.notna(value) and str(value).strip().casefold() not in {"", "nan", "none"}:
            return str(value).strip()
    return "Unknown"


def _prepare_research_schema(df: pd.DataFrame) -> pd.DataFrame:
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
        "Evidence_Confidence": np.nan, "Therapy_Match_Strength": np.nan,
        "Assay_Quality": np.nan, "LOD_Margin": np.nan, "Replicate_Confidence": np.nan,
        "Treatment_Start_Day": 0.0, "Followup_Measurement_Day": np.nan, "Followup_Days": np.nan,
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
    d = source_df.copy()
    design = pd.DataFrame(index=d.index)

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
    design = build_cox_features(source_df, include_kinetics=include_kinetics)
    pfs = safe_numeric_column(source_df, 'PFS_Months', np.nan)
    evt = safe_numeric_column(source_df, 'Progression_Event', np.nan)
    outcome = pd.DataFrame({'PFS_Months': pfs, 'Progression_Event': evt})
    model_df = pd.concat([outcome, design], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

    feature_cols = [c for c in design.columns if c in model_df.columns and model_df[c].nunique(dropna=True) > 1]
    if not feature_cols or model_df['Progression_Event'].sum() < 2:
        return None, model_df, feature_cols

    model_df = model_df[['PFS_Months', 'Progression_Event'] + feature_cols].copy()
    try:
        cph = CoxPHFitter(penalizer=0.08, l1_ratio=0.0)
        cph.fit(model_df, duration_col='PFS_Months', event_col='Progression_Event')
        return cph, model_df, feature_cols
    except Exception:
        try:
            cph = CoxPHFitter()
            cph.fit(model_df, duration_col='PFS_Months', event_col='Progression_Event')
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
    out = source_df.copy()
    mechanistic = safe_numeric_column(out, "Dynamic_CUI", 0.0)
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


st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3022/3022565.png", width=60)
st.sidebar.title("TCF-001 TRACK")
st.sidebar.markdown("---")

data_mode = st.sidebar.radio("Data Source:", ["📂 Upload Custom Cohort", "🔬 Load Validation Demo"])


@st.cache_data
def process_dataframe(df):
    df = _prepare_research_schema(df)
    df["PFS_Months"] = safe_numeric_column(df, "PFS_Months", np.nan)
    df["Progression_Event"] = safe_numeric_column(df, "Progression_Event", np.nan)
    df["Tumor_Fraction"] = safe_numeric_column(df, "Tumor_Fraction", 0.05).clip(lower=0.0001)
    df = df.dropna(subset=["PFS_Months", "Progression_Event"]).copy()

    if "Cohort" not in df.columns: df["Cohort"] = "General Cohort"
    if "Therapy_Type" not in df.columns: df["Therapy_Type"] = "Standard Care"

    for col in ["KRAS_Mutant", "TP53_Mutant", "Therapy_Type", "Cohort"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    for col, default in [("SII", 500.0), ("AST", 25.0), ("ALT", 25.0), ("Age", 50.0), ("Tumor_Fraction_Followup", np.nan)]:
        df[col] = safe_numeric_column(df, col, default)

    df["Evidence_Tier"] = df["Evidence_Tier"].fillna("Unknown")
    df["ESCAT_Tier"] = df["Evidence_Tier"]

    df = split_temporal_roles(df)
    df = derive_feature_frame(df)

    df["VMTB_Matching_Score"] = safe_numeric_column(df, "Dynamic_CUI", 50.0).round(1)
    df["Actionability_Points"] = safe_numeric_column(df, "Evidence_Strength", 0.0) * 100
    df["Therapy_Match"] = safe_numeric_column(df, "Therapy_Compatibility", 0.3)
    df["TP53_Resistance_Base"] = 1.0 - 0.25 * safe_numeric_column(df, "TP53_State", 0.0)
    df["Phi_Host"] = safe_numeric_column(df, "Host_State_Field", 1.0)
    return df


raw_df = None
benchmark_path = os.path.join("data", "Processed_Clinical_Dashboard_Data.xlsx")

if data_mode == "📂 Upload Custom Cohort":
    uploaded_file = st.sidebar.file_uploader("Upload Clinical File (.xlsx, .csv)", type=['xlsx', 'xls', 'csv'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_upload = pd.read_csv(io.BytesIO(uploaded_file.read()))
            else:
                raw_upload = pd.read_excel(io.BytesIO(uploaded_file.read()))
        except Exception as e:
            st.sidebar.error(f"❌ File Parsing Error: {e}")
            st.stop()

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
                    if map_val != "Not Available": rename_dict[map_val] = target

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
        st.sidebar.success("Validation Demo Cohort Loaded.")
    else:
        st.error("Validation file not found in repository (`data/Processed_Clinical_Dashboard_Data.xlsx`).")
        st.stop()

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

baseline_cph, baseline_model_df, baseline_cols = fit_cox_design(filtered_df, include_kinetics=False)
if baseline_cph is not None:
    calibrated_df = compute_empirical_cui(filtered_df, baseline_cph.params_, baseline=50.0)
else:
    calibrated_df = compute_empirical_cui(filtered_df, None)

kinetic_observed = safe_numeric_column(calibrated_df, 'K_Observed', 0.0).sum()
if kinetic_observed >= 8:
    dynamic_cph, dynamic_model_df, dynamic_cols = fit_cox_design(calibrated_df.dropna(subset=['ctDNA_Log_Ratio']), include_kinetics=True)
else:
    dynamic_cph, dynamic_model_df, dynamic_cols = None, pd.DataFrame(), []

if baseline_cph is not None:
    calibrated_df = compute_empirical_cui(calibrated_df, baseline_cph.params_)
else:
    calibrated_df['Calibrated_CUI'] = safe_numeric_column(calibrated_df, 'Dynamic_CUI', 50.0)
    calibrated_df['CUI_Calibration'] = 'Mechanistic fallback (insufficient stable Cox information)'

filtered_df = calibrated_df.copy()

st.title("Decentralized Genomic Profiling & Clinical Analytics")
st.caption("Integrative Host-Tumor Platform: ESCAT, Epistasis, Toxicity (Tau), & Clonal Kinetics")

col1, col2, col3, col4 = st.columns(4)
matched_pts = filtered_df[filtered_df['Therapy_Type'] == 'Targeted/Matched']
matched_rate = round((len(matched_pts) / total_patients) * 100, 1) if total_patients else 0.0

col1.metric("Evaluable Cohort", f"{total_patients:,} pts")
col2.metric("Median PFS", f"{safe_numeric_column(filtered_df, 'PFS_Months', 0.0).median():.1f} Mo")
col3.metric("Targeted Therapy Rate", f"{matched_rate}%")
col4.metric("Mean Clinical Utility (CUI)", f"{safe_numeric_column(filtered_df, 'Calibrated_CUI', 50.0).mean():.1f}")
st.markdown("---")

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

        ax_km.set_title("Survival Probability vs. Baselines", fontsize=12)
        ax_km.set_xlabel("Progression-Free Interval (Months)")
        ax_km.set_ylabel("Probability of PFS $S(t)$")
        ax_km.grid(axis='y', linestyle='--', alpha=0.5)

        st.pyplot(fig_km)
        render_download_button(fig_km, "KM_Benchmark_Overlay", key="km_bench")
        plt.close(fig_km)
    else:
        st.info("Insufficient variance to plot survival curves.")

with tab_cph:
    st.subheader("Multivariable Cox Proportional Hazards Regression")
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
            st.dataframe(summary_table.style.format({'Log-Hazard': '{:.3f}', 'Hazard Ratio': '{:.3f}', 'Std Error': '{:.3f}', 'p-value': '{:.4e}'}), use_container_width=True)

with tab_nomogram:
    st.subheader("Point-of-Care Predictive Nomogram")
    col_n1, col_n2 = st.columns([1, 2])
    with col_n1:
        pt_therapy = st.radio("Therapy Administered", ["Targeted/Matched", "Standard Care"])
        pt_kras = st.selectbox("Actionable Target (KRAS)", ["Yes", "No"])
        pt_tp53 = st.selectbox("Resistance Co-Mutation (TP53)", ["No", "Yes"])
        pt_age = st.slider("Patient Age (Toxicity Modifier)", 30, 90, 50)
        pt_sii = st.slider("Systemic Inflammation (SII)", 200, 1500, 500)
        pt_deritis = st.slider("MASLD Proxy (AST/ALT Ratio)", 0.5, 3.0, 1.0, 0.1)

        patient_row = pd.Series({
            'Therapy_Type': pt_therapy, 'KRAS_Mutant': pt_kras, 'TP53_Mutant': pt_tp53,
            'Age': pt_age, 'SII': pt_sii, 'AST': pt_deritis * 25.0, 'ALT': 25.0, 'Cohort': 'General Cohort'
        })
        patient_features = derive_feature_frame(pd.DataFrame([patient_row]))
        st.metric("Baseline Mechanistic CUI", f"{float(safe_numeric_column(patient_features, 'Baseline_CUI', 50.0).iloc[0]):.1f}")
    with col_n2:
        try:
            if baseline_cph is None or len(baseline_cols) == 0:
                st.info("Insufficient variance.")
            else:
                tmp = pd.DataFrame([{'Therapy_Type': pt_therapy, 'KRAS_Mutant': pt_kras, 'TP53_Mutant': pt_tp53, 'SII': pt_sii, 'DeRitis': pt_deritis, 'Age': pt_age}])
                tmp['Is_Matched'] = (tmp['Therapy_Type'] == 'Targeted/Matched').astype(float)
                tmp['KRAS_Mut'] = (tmp['KRAS_Mutant'] == 'Yes').astype(float)
                tmp['TP53_Mut'] = (tmp['TP53_Mutant'] == 'Yes').astype(float)
                tmp['KRAS_x_TP53'] = tmp['KRAS_Mut'] * tmp['TP53_Mut']
                tmp['Matched_x_TP53'] = tmp['Is_Matched'] * tmp['TP53_Mut']
                tmp['SII_z'] = (tmp['SII'] - filtered_df['SII'].mean()) / (filtered_df['SII'].std(ddof=0) or 1.0)
                tmp['DeRitis_z'] = (tmp['DeRitis'] - filtered_df['DeRitis'].mean()) / (filtered_df['DeRitis'].std(ddof=0) or 1.0)
                tmp['Age_z'] = (tmp['Age'] - filtered_df['Age'].mean()) / (filtered_df['Age'].std(ddof=0) or 1.0)
                for c in baseline_cols:
                    if c not in tmp.columns: tmp[c] = 0.0
                pt_survival = baseline_cph.predict_survival_function(tmp[baseline_cols].astype(float))
                fig_nomo, ax_nomo = plt.subplots(figsize=(7, 4))
                ax_nomo.plot(pt_survival.index, pt_survival.iloc[:, 0], color='darkgreen', linewidth=2.5)
                ax_nomo.set_title("Predicted Individual Survival Trajectory $S(t | Z)$")
                st.pyplot(fig_nomo)
                render_download_button(fig_nomo, "Patient_Survival_Nomogram", key="nomo")
                plt.close(fig_nomo)
        except Exception as e:
            st.warning(f"Error: {e}")

with tab_vmtb:
    st.subheader("Integrative Actionability Distribution (CUI)")
    fig_match = px.histogram(filtered_df, x='Calibrated_CUI', color='ESCAT_Tier', nbins=20, barmode='stack', title="CUI by ESCAT Tier")
    st.plotly_chart(fig_match, use_container_width=True)

with tab_mrd:
    st.subheader("Liquid Biopsy: Circulating Tumor Fraction vs PFS")
    fig_mrd = px.scatter(filtered_df, x='PFS_Months', y='Tumor_Fraction', color='Calibrated_CUI', log_y=True, title="Tumor Fraction vs PFS")
    st.plotly_chart(fig_mrd, use_container_width=True)

with tab_mut:
    st.subheader("Variant Co-occurrence")
    if 'TP53_Mutant' in filtered_df.columns and 'KRAS_Mutant' in filtered_df.columns:
        contingency = pd.crosstab(filtered_df['TP53_Mutant'], filtered_df['KRAS_Mutant'])
        st.write(contingency)

with tab_pathway:
    st.subheader("Dynamic Clinical Decision & Actionability Pathway")
    st.markdown("Automated algorithmic flowchart mapping genomic tiering through host-tumor modulation.")

with tab_data:
    st.subheader("Cohort Dataset & Table 1 Summary")
    st.dataframe(filtered_df, use_container_width=True)

    tier_counts = filtered_df['ESCAT_Tier'].value_counts() if 'ESCAT_Tier' in filtered_df.columns else pd.Series()
    t12_count = (tier_counts.get('Tier I', 0) + tier_counts.get('Tier II', 0)) if not tier_counts.empty else 0
    t12_pct = round((t12_count / total_patients) * 100, 1) if total_patients else 0.0

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
            f"{t12_pct}%"
        ]
    }
    table_1_df = pd.DataFrame(table_1_data)
    st.table(table_1_df)

    st.markdown("#### TC-KUO v3 Research Architecture & Provenance")
    st.json(model_metadata(DEFAULT_PARAMETERS))
    st.caption("The export retains parameter fingerprinting, temporal provenance, observed-component fraction, measurement quality, and separate baseline/dynamic utility fields.")

    csv_buffer = io.BytesIO()
    table_1_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Table 1 (CSV)", data=csv_buffer.getvalue(), file_name="Table1_Baseline_Characteristics.csv", mime="text/csv")

    cohort_csv = io.BytesIO()
    filtered_df.to_csv(cohort_csv, index=False)
    st.download_button("📥 Download Filtered Cohort Dataset (CSV)", data=cohort_csv.getvalue(), file_name="Filtered_Cohort_Analysis.csv", mime="text/csv")
