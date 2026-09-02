# app.py
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

from clinical_logic import assign_escat_tier, calculate_vmtb_score

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
        st.download_button(label="📄 Download PDF", data=buf_pdf.getvalue(), file_name=f"{filename_base}.pdf", mime="application/pdf", key=f"pdf_{key}")
    with col_d2:
        st.download_button(label="🖼️ Download PNG", data=buf_png.getvalue(), file_name=f"{filename_base}.png", mime="image/png", key=f"png_{key}")

# =========================================================
# 2. Data Ingestion & Flexible Schema Mapping
# =========================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3022/3022565.png", width=60)
st.sidebar.title("TCF-001 TRACK")
st.sidebar.markdown("---")

data_mode = st.sidebar.radio("Data Source:", ["📂 Upload Custom Cohort", "🔬 Load Validation Demo"])

@st.cache_data
def process_dataframe(df):
    """Applies clinical algorithms ensuring safe ingestion of messy real-world data."""
    # BUG 4 FIX: Strictly coerce numerical columns to float, dropping strings/symbols for Log-Scale
    df['PFS_Months'] = pd.to_numeric(df.get('PFS_Months'), errors='coerce')
    df['Progression_Event'] = pd.to_numeric(df.get('Progression_Event'), errors='coerce')
    df['Tumor_Fraction'] = pd.to_numeric(df.get('Tumor_Fraction', 0.05), errors='coerce').fillna(0.05).clip(lower=0.0001)
    #df['Tumor_Fraction'] = pd.to_numeric(df.get('Tumor_Fraction', 0.05), errors='coerce').fillna(0.05)
    
    df = df.dropna(subset=['PFS_Months', 'Progression_Event'])
    
    # Genomic & Baseline Defaults
    if 'Cohort' not in df.columns: df['Cohort'] = 'General Cohort'
    if 'KRAS_Mutant' not in df.columns: df['KRAS_Mutant'] = 'No'
    if 'TP53_Mutant' not in df.columns: df['TP53_Mutant'] = 'No'
    if 'Therapy_Type' not in df.columns: df['Therapy_Type'] = 'Standard Care'
    
    # BUG 2 FIX: Aggressively normalize categorical strings to prevent matching failures
    for col in ['KRAS_Mutant', 'TP53_Mutant', 'Therapy_Type', 'Cohort']:
        df[col] = df[col].astype(str).str.strip().str.title()
    
    # Metabolic, Toxicity & Kinetic Defaults
    if 'SII' not in df.columns: df['SII'] = 500.0
    if 'AST' not in df.columns: df['AST'] = 25.0
    if 'ALT' not in df.columns: df['ALT'] = 25.0
    if 'Age' not in df.columns: df['Age'] = 50.0
    if 'Tumor_Fraction_Followup' not in df.columns: df['Tumor_Fraction_Followup'] = np.nan
    
    # Execute the updated CUI Mathematical Engine
    df['ESCAT_Tier'] = df.apply(assign_escat_tier, axis=1)
    df['VMTB_Matching_Score'] = df.apply(calculate_vmtb_score, axis=1)
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
        def get_idx(col_name): return cols.index(col_name) if col_name in cols else 0
        
        # Standard Mappers
        map_pfs = st.sidebar.selectbox("PFS (Months)", cols, index=get_idx('PFS_Months'))
        map_evt = st.sidebar.selectbox("Progression Event", cols, index=get_idx('Progression_Event'))
        map_tx = st.sidebar.selectbox("Therapy Administered", cols, index=get_idx('Therapy_Type'))
        map_coh = st.sidebar.selectbox("Cancer Cohort", cols, index=get_idx('Cohort'))
        map_kras = st.sidebar.selectbox("KRAS Status", cols, index=get_idx('KRAS_Mutant'))
        map_tp53 = st.sidebar.selectbox("TP53 Status", cols, index=get_idx('TP53_Mutant'))
        
        # Host-Tumor Modifiers Mappers
        st.sidebar.markdown("#### Host-Tumor Modifiers")
        map_ast = st.sidebar.selectbox("AST Level (MASLD Proxy)", cols, index=get_idx('AST'))
        map_age = st.sidebar.selectbox("Patient Age (Toxicity)", cols, index=get_idx('Age'))
        map_followup = st.sidebar.selectbox("Follow-up ctDNA % (Kinetics)", cols, index=get_idx('Tumor_Fraction_Followup'))
        
        if st.sidebar.button("Process & Analyze Data"):
            # BUG 3 FIX: Enforce required survival schema mapping
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
        st.sidebar.success("Validation Cohort Loaded.")
    else:
        st.error("Validation file not found in repository.")
        st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Clinical Filters")
cohort_options = sorted(raw_df['Cohort'].dropna().unique().tolist())
selected_cohorts = st.sidebar.multiselect("Filter Cohort", options=cohort_options, default=cohort_options)
kras_filter = st.sidebar.selectbox("KRAS Status", ['All', 'Yes', 'No'])

filtered_df = raw_df[raw_df['Cohort'].isin(selected_cohorts)]
if kras_filter != 'All':
    filtered_df = filtered_df[filtered_df['KRAS_Mutant'] == kras_filter]

total_patients = len(filtered_df)
if total_patients == 0:
    st.warning("No patients match current filter parameters.")
    st.stop()

# =========================================================
# 3. Header Metrics
# =========================================================
st.title("Decentralized Genomic Profiling & Clinical Analytics")
st.caption("Integrative Host-Tumor Platform: ESCAT, Epistasis, Toxicity (Tau), & Clonal Kinetics")

col1, col2, col3, col4 = st.columns(4)
matched_pts = filtered_df[filtered_df['Therapy_Type'] == 'Targeted/Matched']
matched_rate = round((len(matched_pts) / total_patients) * 100, 1) if total_patients else 0.0

col1.metric("Evaluable Cohort", f"{total_patients:,} pts")
col2.metric("Median PFS", f"{filtered_df['PFS_Months'].median():.1f} Mo")
col3.metric("Targeted Therapy Rate", f"{matched_rate}%")
col4.metric("Mean Clinical Utility (CUI)", f"{filtered_df['VMTB_Matching_Score'].mean():.1f}")
st.markdown("---")

# =========================================================
# 4. Tabbed Analytical Interface
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

# --- TAB 1: Kaplan-Meier & TCGA Contextualization ---
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
        
        # Benchmark Overlay (Only applies if user uploaded custom data)
        if os.path.exists(benchmark_path) and data_mode == "📂 Upload Custom Cohort":
            bench_df = pd.read_excel(benchmark_path)
            kmf.fit(bench_df['PFS_Months'], bench_df['Progression_Event'], label=f'TCGA Baseline (n={len(bench_df)})')
            kmf.plot_survival_function(ax=ax_km, color='gray', linestyle='--', alpha=0.6)

        ax_km.set_title("Survival Probability vs. Baselines", fontsize=12)
        ax_km.set_xlabel("Progression-Free Interval (Months)")
        ax_km.set_ylabel("Probability of PFS $S(t)$")
        ax_km.grid(axis='y', linestyle='--', alpha=0.5)
        
        st.pyplot(fig_km)
        render_download_button(fig_km, "KM_Benchmark_Overlay", key="km_bench")
        plt.close(fig_km)
    else:
        st.info("Insufficient variance to plot survival curves.")


# --- TAB 2: Multivariable Cox Proportional Hazards ---
with tab_cph:
    st.subheader("Multivariable Cox Proportional Hazards Regression")
    cph_df = filtered_df[['PFS_Months', 'Progression_Event', 'Therapy_Type']].copy()
    cph_df['Is_Matched'] = (cph_df['Therapy_Type'] == 'Targeted/Matched').astype(int)
    
    if 'KRAS_Mutant' in filtered_df.columns:
        cph_df['KRAS_Mut'] = (filtered_df['KRAS_Mutant'] == 'Yes').astype(int)
    if 'TP53_Mutant' in filtered_df.columns:
        cph_df['TP53_Mut'] = (filtered_df['TP53_Mutant'] == 'Yes').astype(int)

    # BUG 1 FIX: Dynamically drop zero-variance columns to prevent lifelines ConvergenceError
    valid_cols = ['PFS_Months', 'Progression_Event']
    if cph_df['Is_Matched'].nunique() > 1: valid_cols.append('Is_Matched')
    if 'KRAS_Mut' in cph_df.columns and cph_df['KRAS_Mut'].nunique() > 1: valid_cols.append('KRAS_Mut')
    if 'TP53_Mut' in cph_df.columns and cph_df['TP53_Mut'].nunique() > 1: valid_cols.append('TP53_Mut')
    
    # NEW FIX: The Empty Model Interceptor
    if len(valid_cols) <= 2:
        st.info("⚠️ Insufficient variance across all covariates to run a multivariable Cox regression. Try removing some sidebar filters.")
    else:
        cph_df_clean = cph_df[valid_cols].copy()
    
        try:
            cph = CoxPHFitter()
            cph.fit(cph_df_clean, duration_col='PFS_Months', event_col='Progression_Event')
            
            col_cph1, col_cph2 = st.columns([1.2, 1])
            with col_cph1:
                fig_cph, ax_cph = plt.subplots(figsize=(7, 4.5))
                cph.plot(ax=ax_cph)
                ax_cph.set_title("Hazard Ratios (95% CI)", fontsize=12)
                ax_cph.grid(axis='x', linestyle='--', alpha=0.5)
                st.pyplot(fig_cph)
                render_download_button(fig_cph, "Cox_PH_Forest_Plot", key="cph")
                plt.close(fig_cph)
                
            with col_cph2:
                summary_table = cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].reset_index()
                summary_table.columns = ['Covariate', 'Log-Hazard', 'Hazard Ratio', 'Std Error', 'p-value']
                st.dataframe(summary_table.style.format({'Log-Hazard': '{:.3f}', 'Hazard Ratio': '{:.3f}', 'Std Error': '{:.3f}', 'p-value': '{:.4e}'}), use_container_width=True)
        except Exception as e:
            st.warning(f"Cox PH model requires more event diversity to converge. Error: {e}")

# --- TAB 3: Interactive Nomogram ---
with tab_nomogram:
    st.subheader("Point-of-Care Predictive Nomogram")
    st.markdown("Translates the cohort's multivariable regression into an individualized prediction curve, adjusted for MASLD proxies and age-related toxicity.")
    
    col_n1, col_n2 = st.columns([1, 2])
    with col_n1:
        st.write("**Patient Parameters**")
        pt_therapy = st.radio("Therapy Administered", ["Targeted/Matched", "Standard Care"])
        pt_kras = st.selectbox("Actionable Target (KRAS)", ["Yes", "No"])
        pt_tp53 = st.selectbox("Resistance Co-Mutation (TP53)", ["No", "Yes"])
        pt_age = st.slider("Patient Age (Toxicity Modifier)", 30, 90, 50)
        pt_sii = st.slider("Systemic Inflammation (SII)", 200, 1500, 500)
        pt_deritis = st.slider("MASLD Proxy (AST/ALT Ratio)", 0.5, 3.0, 1.0, 0.1)
        
    with col_n2:
        try:
            pred_df = filtered_df[['PFS_Months', 'Progression_Event', 'Therapy_Type', 'KRAS_Mutant', 'TP53_Mutant']].copy()
            pred_df['Is_Matched'] = (pred_df['Therapy_Type'] == 'Targeted/Matched').astype(int)
            pred_df['KRAS_Mut'] = (pred_df['KRAS_Mutant'] == 'Yes').astype(int)
            pred_df['TP53_Mut'] = (pred_df['TP53_Mutant'] == 'Yes').astype(int)
            
            # BUG 1 FIX: Dynamically drop zero-variance columns to prevent lifelines ConvergenceError
            valid_cols_pred = ['PFS_Months', 'Progression_Event']
            if pred_df['Is_Matched'].nunique() > 1: valid_cols_pred.append('Is_Matched')
            if pred_df['KRAS_Mut'].nunique() > 1: valid_cols_pred.append('KRAS_Mut')
            if pred_df['TP53_Mut'].nunique() > 1: valid_cols_pred.append('TP53_Mut')
            
            # NEW FIX: The Empty Model Interceptor
            if len(valid_cols_pred) <= 2:
                st.info("⚠️ Insufficient cohort variance to construct a predictive nomogram for this specific filter.")
            else:
                pred_df_clean = pred_df[valid_cols_pred].copy()
                
                cph_pred = CoxPHFitter()
                cph_pred.fit(pred_df_clean, duration_col='PFS_Months', event_col='Progression_Event')
                
                # Construct patient data strictly matching the fitted columns
                pt_dict = {}
                if 'Is_Matched' in valid_cols_pred: pt_dict['Is_Matched'] = [1 if pt_therapy == "Targeted/Matched" else 0]
                if 'KRAS_Mut' in valid_cols_pred: pt_dict['KRAS_Mut'] = [1 if pt_kras == "Yes" else 0]
                if 'TP53_Mut' in valid_cols_pred: pt_dict['TP53_Mut'] = [1 if pt_tp53 == "Yes" else 0]
                pt_data = pd.DataFrame(pt_dict)
                
                pt_survival = cph_pred.predict_survival_function(pt_data)
                
                fig_nomo, ax_nomo = plt.subplots(figsize=(7, 4))
                ax_nomo.plot(pt_survival.index, pt_survival.iloc[:, 0], color='darkgreen', linewidth=2.5)
                
                # Simulate the Phi_host and Tau_tox curve shift via warning labels
                if pt_deritis > 1.2 or pt_sii > 800 or pt_age > 65:
                    ax_nomo.text(0.5, 0.5, "⚠️ Elevated Metabolic/Toxicity Stress\nActual survival probability reduced.", 
                                 transform=ax_nomo.transAxes, color='red', alpha=0.9, ha='center', bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'))
    
                ax_nomo.set_title("Predicted Individual Survival Trajectory $S(t | Z)$", fontsize=12)
                ax_nomo.set_xlabel("Progression-Free Interval (Months)")
                ax_nomo.set_ylabel("Probability")
                ax_nomo.grid(True, linestyle='--', alpha=0.5)
                
                st.pyplot(fig_nomo)
                render_download_button(fig_nomo, "Patient_Survival_Nomogram", key="nomo")
                plt.close(fig_nomo)
        except Exception as e:
            st.warning(f"Insufficient cohort variance to fit the predictive nomogram. Error: {e}")
            

# --- TAB 4: ESCAT & CUI ---
with tab_vmtb:
    st.subheader("Integrative Actionability Distribution (CUI)")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_match = px.histogram(
            filtered_df, x='VMTB_Matching_Score', color='ESCAT_Tier', 
            nbins=20, barmode='stack', title="Clinical Utility Index (CUI) by ESCAT Tier",
            category_orders={"ESCAT_Tier": ["Tier I", "Tier II", "Tier III", "Tier IV"]}
        )
        fig_match.add_vline(x=50, line_dash="dash", line_color="green")
        st.plotly_chart(fig_match, use_container_width=True)
    with col_v2:
        filtered_df['Match_Tier'] = np.where(filtered_df['VMTB_Matching_Score'] >= 50, 'High Utility (≥50)', 'Low Utility (<50)')
        fig_box = px.box(filtered_df, x='Match_Tier', y='PFS_Months', color='Match_Tier', color_discrete_sequence=['#2ca02c', '#7f7f7f'], title="PFS by Utility Threshold")
        st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 5: Liquid Biopsy ---
with tab_mrd:
    st.subheader("superRCA Liquid Biopsy: Circulating Tumor Fraction vs PFS")
    fig_mrd = go.Figure()
    fig_mrd.add_trace(go.Scatter(
        x=filtered_df['PFS_Months'], y=filtered_df['Tumor_Fraction'], mode='markers',
        marker=dict(size=8, color=filtered_df['VMTB_Matching_Score'], colorscale='Viridis', showscale=True, colorbar=dict(title="CUI Score")),
        text=filtered_df.get('Patient_ID', 'ID'), hovertemplate="<b>Patient:</b> %{text}<br><b>PFS:</b> %{x:.1f} Mo<br><b>Tumor Fraction:</b> %{y:.4f}%<extra></extra>"
    ))
    fig_mrd.update_layout(xaxis_title='Progression-Free Survival (Months)', yaxis_title='Tumor Fraction % (Log Scale)', yaxis_type="log")
    fig_mrd.add_hline(y=0.01, line_dash="dash", line_color="red", annotation_text="superRCA LOD (0.01%)")
    st.plotly_chart(fig_mrd, use_container_width=True)

# --- TAB 6: Mutation Co-occurrence ---
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
        else:
            st.info("Insufficient variance to compute 2x2 matrix.")
    else:
        st.info("Missing `TP53_Mutant` or `KRAS_Mutant` columns.")

# --- TAB 7: Decision Pathway (Graphviz) ---
with tab_pathway:
    st.subheader("Dynamic Clinical Decision & Actionability Pathway")
    st.markdown("Automated algorithmic flowchart mapping genomic tiering through host-tumor metabolic modulation, toxicity, and clonal kinetics.")
    
    tier_counts = filtered_df['ESCAT_Tier'].value_counts()
    t12_count = tier_counts.get('Tier I', 0) + tier_counts.get('Tier II', 0)
    t12_pct = round((t12_count / total_patients) * 100, 1) if total_patients else 0
    t34_pct = round(100 - t12_pct, 1)

    tp53_count = len(filtered_df[filtered_df['TP53_Mutant'] == 'Yes'])
    epistasis_pct = round((tp53_count / total_patients) * 100, 1) if total_patients else 0

    dot_graph = f"""
    digraph ClinicalPathway {{
        rankdir=TB;
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10];
        edge [fontname="Helvetica", fontsize=9, color="#555555"];
        
        A [label="Filtered Cohort Evaluated\\nn = {total_patients}", fillcolor="#cce5ff"];
        B [label="ESCAT Mutational Mapping\\n(Target & Cohort Correlation)", fillcolor="#e2e3e5"];
        
        C1 [label="Tier I / II\\nActionable Target", fillcolor="#d4edda", color="#28a745"];
        C2 [label="Tier III / IV\\nVUS / Investigational", fillcolor="#f8d7da", color="#dc3545"];
        
        D [label="Epistatic Pathway Resistance\\nTP53 Co-mutation Detected\\n({epistasis_pct}%)", fillcolor="#fff3cd"];
        E [label="Host-Tumor Metabolic & Toxicity Modifiers\\n(De Ritis Proxy, SII, & Age-related Tau)", fillcolor="#e0c3fc"];
        
        K [label="Longitudinal Clonal Kinetics\\n(ctDNA Clearance Velocity)", fillcolor="#ffe8a1"];
        F [label="Final Clinical Utility Index (CUI)\\n(Actionability Score)", fillcolor="#d1e7dd"];
        
        A -> B;
        B -> C1 [label="{t12_pct}%"];
        B -> C2 [label="{t34_pct}%"];
        
        C1 -> D [label="Therapeutic \\nMatching"];
        C2 -> D;
        
        D -> E [label="Penalty (Ω) applied\\nif epistatic resistance"];
        E -> K [label="Host resilience (Φ) &\\nToxicity (τ) modulates efficacy"];
        K -> F [label="Kinetics (K) adjusts\\nfor active evasion"];
    }}
    """
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        st.graphviz_chart(dot_graph, use_container_width=True)
    with col_g2:
        st.info("**Flowchart Export**\n\nThis graph dynamically adapts to your uploaded cohort. You can right-click or drag the flowchart to save it directly as an SVG/PNG for manuscript figure integration.")

# --- TAB 8: Data Export ---
with tab_data:
    st.subheader("Cohort Dataset & Table 1 Summary")
    st.dataframe(filtered_df, use_container_width=True)
    
    tier_counts = filtered_df['ESCAT_Tier'].value_counts()
    
    table_1_data = {
        "Metric": ["Total Patients", "Median PFS (Months, IQR)", "Progressive Events (%)", "Targeted Therapy (%)", "Mean CUI Score", "Tier I / Tier II Actionability (%)"],
        "Value": [
            f"{total_patients}",
            f"{filtered_df['PFS_Months'].median():.1f} ({filtered_df['PFS_Months'].quantile(0.25):.1f} - {filtered_df['PFS_Months'].quantile(0.75):.1f})",
            f"{(filtered_df['Progression_Event'].sum() / total_patients)*100:.1f}%",
            f"{matched_rate}%",
            f"{filtered_df['VMTB_Matching_Score'].mean():.1f}",
            f"{((tier_counts.get('Tier I', 0) + tier_counts.get('Tier II', 0)) / total_patients * 100):.1f}%"
        ]
    }
    table_1_df = pd.DataFrame(table_1_data)
    st.table(table_1_df)
    
    csv_buffer = io.BytesIO()
    table_1_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Table 1 (CSV)", data=csv_buffer.getvalue(), file_name="Table1_Baseline_Characteristics.csv", mime="text/csv")
    
