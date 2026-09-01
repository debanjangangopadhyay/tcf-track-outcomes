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

# =========================================================
# 1. Page Configuration & Memory-Safe Styling
# =========================================================
st.set_page_config(
    page_title="TCF-001 TRACK / Precision Oncology Analytics",
    page_icon="🧬",
    layout="wide"
)

# Apply Clean Academic Matplotlib Theme
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def render_download_button(fig, filename_base: str, key: str):
    """Encodes matplotlib figure directly into in-memory PDF/PNG buffers."""
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
# 2. VMTB Clinical Logic Formula (ESCAT + superRCA)
# =========================================================
def calculate_vmtb_score(row):
    """
    Deterministic VMTB matching score based on ESCAT tiers, 
    superRCA ctDNA confidence, and therapeutic administration.
    """
    # 1. ESCAT Evidence Tier Base Score (E_tier)
    tier = str(row.get('ESCAT_Tier', 'Tier I')).strip()
    if tier == 'Tier I': e_tier = 100.0
    elif tier == 'Tier II': e_tier = 75.0
    elif tier == 'Tier III': e_tier = 50.0
    else: e_tier = 25.0  # VUS / Tier IV
        
    # 2. superRCA Confidence Modifier (C_ctDNA) (LOD = 0.01%)
    tumor_fraction = float(row.get('Tumor_Fraction', 0.05))
    c_ctdna = 1.0 if tumor_fraction >= 0.01 else 0.70
    
    # 3. Therapeutic Match Modifier (M_tx)
    therapy = str(row.get('Therapy_Type', 'Standard Care')).strip()
    m_tx = 1.0 if therapy == 'Targeted/Matched' else 0.30
    
    final_score = e_tier * c_ctdna * m_tx
    return min(100.0, round(final_score, 1))

# =========================================================
# 3. Data Ingestion Pipeline (Real-World Data)
# =========================================================
@st.cache_data
def load_real_world_data():
    """Loads the cBioPortal derived dataset directly from the data/ folder."""
    file_path = os.path.join("data", "Processed_Clinical_Dashboard_Data.xlsx")
    fallback_path = "Processed_Clinical_Dashboard_Data.xlsx" # If placed in root
    
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    elif os.path.exists(fallback_path):
        df = pd.read_excel(fallback_path)
    else:
        st.error("🚨 Critical Error: `Processed_Clinical_Dashboard_Data.xlsx` not found. Please ensure the file is placed inside a `data/` folder in your GitHub repository.")
        st.stop()
        
    # Coerce critical analytical datatypes
    df['PFS_Months'] = pd.to_numeric(df['PFS_Months'], errors='coerce')
    df['Progression_Event'] = pd.to_numeric(df['Progression_Event'], errors='coerce')
    df = df.dropna(subset=['PFS_Months', 'Progression_Event'])
    
    # Ensure optional clinical fields have fallbacks to prevent crashes
    if 'Tumor_Fraction' not in df.columns: df['Tumor_Fraction'] = 0.05
    if 'Cohort' not in df.columns: df['Cohort'] = 'General Cohort'
    if 'KRAS_Mutant' not in df.columns: df['KRAS_Mutant'] = 'No'
    if 'TP53_Mutant' not in df.columns: df['TP53_Mutant'] = 'No'
    
    # Automatically calculate the scientific VMTB score
    df['VMTB_Matching_Score'] = df.apply(calculate_vmtb_score, axis=1)
    
    return df

raw_df = load_real_world_data()

# =========================================================
# 4. Sidebar Dynamic Filtering
# =========================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3022/3022565.png", width=60) # Simple DNA icon
st.sidebar.title("TCF-001 TRACK")
st.sidebar.success("✅ Real-World Validation Data Loaded")
st.sidebar.markdown("---")
st.sidebar.header("Clinical Filters")

cohort_options = sorted(raw_df['Cohort'].dropna().unique().tolist())
selected_cohorts = st.sidebar.multiselect("Filter Cohort", options=cohort_options, default=cohort_options)
kras_filter = st.sidebar.selectbox("KRAS Status", ['All', 'Yes', 'No'])

filtered_df = raw_df[raw_df['Cohort'].isin(selected_cohorts)]
if kras_filter != 'All':
    filtered_df = filtered_df[filtered_df['KRAS_Mutant'] == kras_filter]

total_patients = len(filtered_df)

if total_patients == 0:
    st.warning("No patients match current filter parameters. Broaden your sidebar selections.")
    st.stop()

# =========================================================
# 5. Header & Executive Summary Metrics
# =========================================================
st.title("Decentralized Genomic Profiling & Clinical Analytics")
st.caption("Precision Oncology Platform: Kaplan-Meier Survival, Cox PH Regression, ESCAT/VMTB Matching & superRCA Liquid Biopsy")

col1, col2, col3, col4 = st.columns(4)
matched_pts = filtered_df[filtered_df['Therapy_Type'] == 'Targeted/Matched']
matched_rate = round((len(matched_pts) / total_patients) * 100, 1) if total_patients else 0.0

col1.metric("Evaluable Cohort", f"{total_patients:,} pts")
col2.metric("Median PFS", f"{filtered_df['PFS_Months'].median():.1f} Mo")
col3.metric("Targeted Therapy Rate", f"{matched_rate}%")
col4.metric("Mean VMTB Score", f"{filtered_df['VMTB_Matching_Score'].mean():.1f}")
st.markdown("---")

# =========================================================
# 6. Tabbed Analytical Interface
# =========================================================
tab_surv, tab_cph, tab_vmtb, tab_mrd, tab_mut, tab_data = st.tabs([
    "📈 Survival (Kaplan-Meier)",
    "🌲 Multivariable Cox PH",
    "🎯 VMTB Actionability",
    "🔬 superRCA Liquid Biopsy",
    "🧬 Mutation Co-Occurrence",
    "📋 Data Export (Table 1)"
])

# ---------------------------------------------------------
# TAB 1: Kaplan-Meier Survival Analysis
# ---------------------------------------------------------
with tab_surv:
    st.subheader("Kaplan-Meier Progression-Free Survival Analysis")
    
    matched = filtered_df[filtered_df['Therapy_Type'] == 'Targeted/Matched']
    unmatched = filtered_df[filtered_df['Therapy_Type'] == 'Standard Care']
    
    fig_km, ax_km = plt.subplots(figsize=(8, 4.8))
    kmf = KaplanMeierFitter()
    
    if len(matched) > 0 and len(unmatched) > 0:
        kmf.fit(matched['PFS_Months'], matched['Progression_Event'], label=f'Targeted / Matched (n={len(matched)})')
        kmf.plot_survival_function(ax=ax_km, color='#1f77b4', ci_show=True, lw=2)
        
        kmf.fit(unmatched['PFS_Months'], unmatched['Progression_Event'], label=f'Standard Care (n={len(unmatched)})')
        kmf.plot_survival_function(ax=ax_km, color='#d62728', ci_show=True, lw=2)
        
        lr_result = logrank_test(matched['PFS_Months'], unmatched['PFS_Months'], matched['Progression_Event'], unmatched['Progression_Event'])
        
        ax_km.set_title("Progression-Free Survival by Therapeutic Arm", fontsize=12, pad=10)
        ax_km.set_xlabel("Progression-Free Interval (Months)", fontsize=10)
        ax_km.set_ylabel("Probability of Progression-Free Survival $S(t)$", fontsize=10)
        ax_km.grid(axis='y', linestyle='--', alpha=0.5)
        
        p_val_text = f"p < 0.0001" if lr_result.p_value < 0.0001 else f"p = {lr_result.p_value:.4f}"
        ax_km.text(0.62, 0.82, f"Log-rank {p_val_text}", transform=ax_km.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#ccc'))
        st.pyplot(fig_km)
        render_download_button(fig_km, "Kaplan_Meier_Survival_Curve", key="km")
        plt.close(fig_km)
    else:
        st.info("Insufficient variance in therapeutic arms to plot survival curves.")

# ---------------------------------------------------------
# TAB 2: Multivariable Cox Proportional Hazards Model
# ---------------------------------------------------------
with tab_cph:
    st.subheader("Multivariable Cox Proportional Hazards Regression")
    
    cph_df = filtered_df[['PFS_Months', 'Progression_Event', 'Therapy_Type']].copy()
    cph_df['Is_Matched'] = (cph_df['Therapy_Type'] == 'Targeted/Matched').astype(int)
    cph_df.drop(columns=['Therapy_Type'], inplace=True)
    
    if 'Age' in filtered_df.columns:
        cph_df['Age'] = pd.to_numeric(filtered_df['Age'], errors='coerce').fillna(filtered_df['Age'].median())
    if 'KRAS_Mutant' in filtered_df.columns:
        cph_df['KRAS_Mut'] = (filtered_df['KRAS_Mutant'] == 'Yes').astype(int)
    if 'TP53_Mutant' in filtered_df.columns:
        cph_df['TP53_Mut'] = (filtered_df['TP53_Mutant'] == 'Yes').astype(int)

    try:
        cph = CoxPHFitter()
        cph.fit(cph_df, duration_col='PFS_Months', event_col='Progression_Event')
        
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
            summary_table.columns = ['Covariate', 'Log-Hazard', 'Hazard Ratio (HR)', 'Std Error', 'p-value']
            st.dataframe(summary_table.style.format({'Log-Hazard': '{:.3f}', 'Hazard Ratio (HR)': '{:.3f}', 'Std Error': '{:.3f}', 'p-value': '{:.4e}'}), use_container_width=True)
    except Exception as e:
        st.warning(f"Cox PH model requires more event diversity. Error: {e}")

# ---------------------------------------------------------
# TAB 3: VMTB Actionability Analysis
# ---------------------------------------------------------
with tab_vmtb:
    st.subheader("ESCAT-Derived Virtual Molecular Tumor Board (VMTB) Actionability")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_match = px.histogram(filtered_df, x='VMTB_Matching_Score', color='Therapy_Type', nbins=20, barmode='overlay', color_discrete_sequence=['#1f77b4', '#d62728'], title="Distribution of VMTB Scores")
        fig_match.add_vline(x=50, line_dash="dash", line_color="green", annotation_text="High Match Threshold (≥50)")
        st.plotly_chart(fig_match, use_container_width=True)
        
    with col_v2:
        filtered_df['Match_Tier'] = np.where(filtered_df['VMTB_Matching_Score'] >= 50, 'High Match (≥50)', 'Low Match (<50)')
        fig_box = px.box(filtered_df, x='Match_Tier', y='PFS_Months', color='Match_Tier', color_discrete_sequence=['#2ca02c', '#7f7f7f'], title="PFS by VMTB Threshold")
        st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: superRCA Liquid Biopsy & MRD Monitoring
# ---------------------------------------------------------
with tab_mrd:
    st.subheader("superRCA Liquid Biopsy: Circulating Tumor Fraction vs PFS")
    
    fig_mrd = go.Figure()
    fig_mrd.add_trace(go.Scatter(
        x=filtered_df['PFS_Months'], y=filtered_df['Tumor_Fraction'], mode='markers',
        marker=dict(size=8, color=filtered_df['VMTB_Matching_Score'], colorscale='Viridis', showscale=True, colorbar=dict(title="VMTB Score")),
        text=filtered_df.get('Patient_ID', 'ID'), hovertemplate="<b>Patient:</b> %{text}<br><b>PFS:</b> %{x:.1f} Mo<br><b>Tumor Fraction:</b> %{y:.4f}%<extra></extra>"
    ))
    fig_mrd.update_layout(xaxis_title='Progression-Free Survival (Months)', yaxis_title='Circulating Tumor Fraction % (Log Scale)', yaxis_type="log")
    fig_mrd.add_hline(y=0.01, line_dash="dash", line_color="red", annotation_text="superRCA LOD (0.01%)")
    st.plotly_chart(fig_mrd, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: Genomic Variant Co-occurrence (Fisher's Exact Test)
# ---------------------------------------------------------
with tab_mut:
    st.subheader("Variant Co-occurrence & Mutual Exclusivity")
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
            st.info("Insufficient variance to compute a 2x2 contingency matrix.")
    else:
        st.info("Dataset missing `TP53_Mutant` or `KRAS_Mutant` columns.")

# ---------------------------------------------------------
# TAB 6: Data Table & Baseline Characteristics Export
# ---------------------------------------------------------
with tab_data:
    st.subheader("Cohort Dataset & Table 1 Summary")
    st.dataframe(filtered_df, use_container_width=True)
    
    table_1_data = {
        "Metric": ["Total Patients", "Median PFS (Months, IQR)", "Progressive Events (%)", "Targeted Therapy (%)", "Mean VMTB Score", "Median Tumor Fraction (%)"],
        "Value": [
            f"{total_patients}",
            f"{filtered_df['PFS_Months'].median():.1f} ({filtered_df['PFS_Months'].quantile(0.25):.1f} - {filtered_df['PFS_Months'].quantile(0.75):.1f})",
            f"{(filtered_df['Progression_Event'].sum() / total_patients)*100:.1f}%",
            f"{matched_rate}%",
            f"{filtered_df['VMTB_Matching_Score'].mean():.1f}",
            f"{filtered_df['Tumor_Fraction'].median():.4f}%"
        ]
    }
    table_1_df = pd.DataFrame(table_1_data)
    st.table(table_1_df)
    
    csv_buffer = io.BytesIO()
    table_1_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Table 1 (CSV)", data=csv_buffer.getvalue(), file_name="Table1_Baseline_Characteristics.csv", mime="text/csv")
        
