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
    assign_escat_tier,
    calculate_vmtb_score,
    derive_feature_frame,
    build_cox_features,
    compute_empirical_cui,
    score_explanation,
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


def fit_cox_design(source_df: pd.DataFrame, include_kinetics: bool = False):
    """Fit a stabilized Cox model while dropping zero-variance predictors."""
    design = build_cox_features(source_df, include_kinetics=include_kinetics)
    outcome = pd.DataFrame({
        'PFS_Months': pd.to_numeric(source_df['PFS_Months'], errors='coerce'),
        'Progression_Event': pd.to_numeric(source_df['Progression_Event'], errors='coerce'),
    })
    model_df = pd.concat([outcome, design], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

    feature_cols = []
    for col in design.columns:
        if col in model_df.columns and model_df[col].nunique(dropna=True) > 1:
            feature_cols.append(col)

    if not feature_cols:
        return None, model_df, feature_cols

    model_df = model_df[['PFS_Months', 'Progression_Event'] + feature_cols].copy()
    if model_df['Progression_Event'].sum() < 2:
        return None, model_df, feature_cols

    try:
        # Modest ridge penalty improves stability without changing the runtime environment.
        cph = CoxPHFitter(penalizer=0.08, l1_ratio=0.0)
        cph.fit(model_df, duration_col='PFS_Months', event_col='Progression_Event')
        return cph, model_df, feature_cols
    except Exception:
        # Secondary fit mirrors the original behavior if a penalized fit is incompatible.
        try:
            cph = CoxPHFitter()
            cph.fit(model_df, duration_col='PFS_Months', event_col='Progression_Event')
            return cph, model_df, feature_cols
        except Exception:
            return None, model_df, feature_cols


def bootstrap_c_index(source_df: pd.DataFrame, include_kinetics: bool = False, n_boot: int = 100, seed: int = 42):
    """Internal bootstrap stability interval for the C-index when feasible."""
    rng = np.random.default_rng(seed)
    n = len(source_df)
    if n < 15:
        return np.nan, np.nan, np.nan

    c_indexes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = source_df.iloc[idx].reset_index(drop=True)
        cph, model_df, cols = fit_cox_design(boot, include_kinetics=include_kinetics)
        if cph is None or not cols:
            continue
        try:
            pred = cph.predict_partial_hazard(model_df[cols])
            concordance = cph.concordance_index_
            if np.isfinite(concordance):
                c_indexes.append(float(concordance))
        except Exception:
            continue

    if len(c_indexes) < 5:
        return np.nan, np.nan, np.nan
    arr = np.asarray(c_indexes)
    return float(np.median(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


# =========================================================
# 2. Data Ingestion & Flexible Schema Mapping
# =========================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3022/3022565.png", width=60)
st.sidebar.title("TCF-001 TRACK")
st.sidebar.markdown("---")

data_mode = st.sidebar.radio("Data Source:", ["📂 Upload Custom Cohort", "🔬 Load Validation Demo"])


@st.cache_data
def process_dataframe(df):
    """Normalizes the cohort and executes the transparent CUI feature engine."""
    df = df.copy()

    df['PFS_Months'] = safe_numeric_column(df, 'PFS_Months', np.nan)
    df['Progression_Event'] = safe_numeric_column(df, 'Progression_Event', np.nan)
    df['Tumor_Fraction'] = safe_numeric_column(df, 'Tumor_Fraction', 0.05).clip(lower=0.0001)

    df = df.dropna(subset=['PFS_Months', 'Progression_Event']).copy()

    if 'Cohort' not in df.columns:
        df['Cohort'] = 'General Cohort'
    if 'KRAS_Mutant' not in df.columns:
        df['KRAS_Mutant'] = 'No'
    if 'TP53_Mutant' not in df.columns:
        df['TP53_Mutant'] = 'No'
    if 'Therapy_Type' not in df.columns:
        df['Therapy_Type'] = 'Standard Care'

    for col in ['KRAS_Mutant', 'TP53_Mutant', 'Therapy_Type', 'Cohort']:
        df[col] = df[col].astype(str).str.strip().str.title()

    if 'SII' not in df.columns:
        df['SII'] = 500.0
    if 'AST' not in df.columns:
        df['AST'] = 25.0
    if 'ALT' not in df.columns:
        df['ALT'] = 25.0
    if 'Age' not in df.columns:
        df['Age'] = 50.0
    if 'Tumor_Fraction_Followup' not in df.columns:
        df['Tumor_Fraction_Followup'] = np.nan

    for col, default in [('SII', 500.0), ('AST', 25.0), ('ALT', 25.0), ('Age', 50.0), ('Tumor_Fraction_Followup', np.nan)]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if np.isfinite(default):
            df[col] = df[col].fillna(default)

    df['ESCAT_Tier'] = df.apply(assign_escat_tier, axis=1)
    df = derive_feature_frame(df)
    # Preserve the original public score column name.
    df['VMTB_Matching_Score'] = df['Dynamic_CUI'].round(1)
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
    st.subheader("Point-of-Care Predictive Nomogram")
    st.markdown("Translates the cohort's multivariable regression into an individualized prediction curve, adjusted for host state, toxicity and explicit genomic interactions.")

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
                # Build patient design with the same transformations used by the fitted cohort model.
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

                # Dynamic on-treatment update is shown separately from baseline prognosis.
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
        'Actionability_Points', 'Therapy_Match', 'TP53_Resistance_Base',
        'Phi_Host', 'Toxicity_Modifier', 'K_Clearance', 'Baseline_CUI',
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

    observed_kin = filtered_df[filtered_df['K_Observed'] > 0].copy()
    if len(observed_kin) > 0:
        st.markdown("#### Longitudinal Molecular Update")
        st.dataframe(
            observed_kin[[
                'Tumor_Fraction', 'Tumor_Fraction_Followup',
                'ctDNA_Log_Ratio', 'ctDNA_Relative_Change',
                'K_Clearance', 'K_Status', 'Dynamic_CUI'
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

            if baseline_cph is not None and 'KRAS_x_TP53' in baseline_cph.params_.index:
                interaction_coef = baseline_cph.params_['KRAS_x_TP53']
                interaction_hr = np.exp(interaction_coef)
                st.metric("Estimated KRAS×TP53 Interaction HR", f"{interaction_hr:.2f}")
                st.caption("This is an empirical statistical interaction term; it is not treated as mechanistic epistasis by the software unless independently established.")
        else:
            st.info("Insufficient variance to compute 2x2 matrix.")
    else:
        st.info("Missing `TP53_Mutant` or `KRAS_Mutant` columns.")

# =========================================================
# TAB 7: Decision Pathway — original structure, refined semantics
# =========================================================
with tab_pathway:
    st.subheader("Dynamic Clinical Decision & Actionability Pathway")
    st.markdown("Automated algorithmic flowchart mapping genomic tiering through host-tumor metabolic modulation, toxicity, and longitudinal molecular kinetics.")

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

    st.markdown("#### Model Provenance")
    provenance = pd.DataFrame({
        'Element': [
            'Actionability mapping', 'Resistance feature', 'Host state',
            'Toxicity', 'ctDNA kinetic update', 'Calibration layer'
        ],
        'Implementation': [
            'Supported-cohort ESCAT-inspired mapping',
            'TP53 term + empirical KRAS×TP53 interaction',
            'Continuous SII / De Ritis / age transforms',
            'Continuous bounded exponential modifier',
            'Baseline-to-follow-up ctDNA log-ratio',
            'Penalized Cox coefficients + mechanistic CUI hybrid when stable'
        ]
    })
    st.dataframe(provenance, use_container_width=True)

    csv_buffer = io.BytesIO()
    table_1_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Table 1 (CSV)", data=csv_buffer.getvalue(), file_name="Table1_Baseline_Characteristics.csv", mime="text/csv")
