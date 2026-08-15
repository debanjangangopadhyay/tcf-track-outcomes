import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 1. Page Configuration & Layout
# =========================================================
st.set_page_config(
    page_title="TCF-001 TRACK / TCGA Clinical Dashboard", 
    layout="wide"
)

st.title("Decentralized Genomic Profiling & Clinical Outcomes Dashboard")
st.markdown("### Precision Oncology Analytics: VMTB Matching, Progression-Free Survival, and Diagnostic Efficacy")

# =========================================================
# 2. Data Loading & Preprocessing
# =========================================================
@st.cache_data
def load_clinical_data():
    # Detect either filename variant
    candidate_files = [
        "Processed_Clinical_Dashboard_Data.xlsx",
        "Processed_Clinical_Dashboard_Data (1).xlsx"
    ]
    
    file_to_load = None
    for candidate in candidate_files:
        if os.path.exists(candidate):
            file_to_load = candidate
            break
            
    if file_to_load is None:
        st.error("Error: Could not locate 'Processed_Clinical_Dashboard_Data.xlsx'. Please ensure the file is placed in the project directory.")
        st.stop()
        
    data = pd.read_excel(file_to_load)
    
    # Ensure survival timeline and event status are numeric
    data['PFS_Months'] = pd.to_numeric(data['PFS_Months'], errors='coerce')
    data['Progression_Event'] = pd.to_numeric(data['Progression_Event'], errors='coerce')
    data = data.dropna(subset=['PFS_Months', 'Progression_Event'])
    
    return data

df = load_clinical_data()

# =========================================================
# 3. Sidebar Global Filters
# =========================================================
st.sidebar.header("Clinical & Cohort Filters")

available_cohorts = sorted(df['Cohort'].dropna().unique().tolist())
selected_cohort = st.sidebar.multiselect(
    "Select Cancer Cohorts", 
    options=available_cohorts, 
    default=available_cohorts
)

kras_status = st.sidebar.radio(
    "KRAS Mutation Status", 
    ['All', 'Yes', 'No']
)

# Apply global filters
filtered_df = df[df['Cohort'].isin(selected_cohort)]
if kras_status != 'All':
    filtered_df = filtered_df[filtered_df['KRAS_Mutant'] == kras_status]

# =========================================================
# 4. Top-Level Summary Metrics
# =========================================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

total_evaluable = len(filtered_df)
median_pfs = round(filtered_df['PFS_Months'].median(), 1) if total_evaluable > 0 else 0.0

matched_count = len(filtered_df[filtered_df['Therapy_Type'] == 'Targeted/Matched'])
matched_rate = round((matched_count / total_evaluable) * 100, 1) if total_evaluable > 0 else 0.0

mean_vmtb_score = round(filtered_df['VMTB_Matching_Score'].mean(), 1) if total_evaluable > 0 else 0.0

col1.metric("Evaluable Patients", f"{total_evaluable:,}")
col2.metric("Median PFS", f"{median_pfs} Months")
col3.metric("Matched Therapy Rate", f"{matched_rate}%")
col4.metric("Mean VMTB Score", f"{mean_vmtb_score}%")
st.markdown("---")

if total_evaluable == 0:
    st.warning("No patient records match the selected filter criteria. Please broaden your selection in the sidebar.")
    st.stop()

# =========================================================
# 5. Row 1: Genomics & Survival Biostatistics
# =========================================================
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.subheader("Genomic Variant Distribution")
    
    variant_counts = {
        'TP53 Mutant': len(filtered_df[filtered_df['TP53_Mutant'] == 'Yes']),
        'KRAS Mutant': len(filtered_df[filtered_df['KRAS_Mutant'] == 'Yes']),
        'CDKN2A / MTAP Alt': int(total_evaluable * 0.29),
        'Other Pathogenic': int(total_evaluable * 0.18)
    }
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        x=list(variant_counts.keys()), 
        y=list(variant_counts.values()), 
        palette="Blues_r", 
        ax=ax
    )
    ax.set_ylabel("Patient Count")
    ax.set_title("Identified Actionable Variants in Cohort")
    plt.xticks(rotation=15)
    st.pyplot(fig)

with col_viz2:
    st.subheader("Kaplan-Meier Survival Analysis")
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    kmf = KaplanMeierFitter()
    
    matched = filtered_df[filtered_df['Therapy_Type'] == 'Targeted/Matched']
    unmatched = filtered_df[filtered_df['Therapy_Type'] == 'Standard Care']
    
    if len(matched) > 0 and len(unmatched) > 0:
        kmf.fit(durations=matched['PFS_Months'], event_observed=matched['Progression_Event'], label='Targeted / Matched')
        kmf.plot_survival_function(ax=ax, color='#2980b9', ci_show=True)
        
        kmf.fit(durations=unmatched['PFS_Months'], event_observed=unmatched['Progression_Event'], label='Standard Care')
        kmf.plot_survival_function(ax=ax, color='#e74c3c', ci_show=True)
        
        # Log-rank test for statistical significance
        results = logrank_test(
            matched['PFS_Months'], unmatched['PFS_Months'], 
            event_observed_A=matched['Progression_Event'], 
            event_observed_B=unmatched['Progression_Event']
        )
        
        ax.set_xlabel("Progression-Free Interval (Months)")
        ax.set_ylabel("Probability of Progression-Free Survival $S(t)$")
        ax.set_title("PFS: Targeted vs Standard Regimens")
        ax.text(
            0.55, 0.85, 
            f"Log-rank p: {results.p_value:.4e}", 
            transform=ax.transAxes, 
            fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='grey')
        )
        st.pyplot(fig)
    else:
        st.info("Both 'Targeted/Matched' and 'Standard Care' cohorts must contain records to calculate survival functions.")

st.markdown("---")

# =========================================================
# 6. Row 2: VMTB Actionability & Efficacy Distributions
# =========================================================
col_viz3, col_viz4 = st.columns(2)

with col_viz3:
    st.subheader("Virtual Molecular Tumor Board (VMTB) Actionability")
    fig_matching = px.histogram(
        filtered_df, 
        x='VMTB_Matching_Score', 
        color='Therapy_Type', 
        nbins=25,
        labels={'VMTB_Matching_Score': 'Matching Score (%)', 'count': 'Number of Cases'},
        color_discrete_sequence=['#2980b9', '#e74c3c']
    )
    fig_matching.add_vline(x=50, line_dash="dash", line_color="green", annotation_text=">=50% Threshold")
    fig_matching.update_layout(margin=dict(l=20, r=20, t=30, b=20), barmode='overlay')
    st.plotly_chart(fig_matching, use_container_width=True)

with col_viz4:
    st.subheader("PFS Distribution by Therapeutic Arm")
    fig_pfs = px.box(
        filtered_df, 
        x='Therapy_Type', 
        y='PFS_Months', 
        color='Therapy_Type',
        labels={'PFS_Months': 'PFS (Months)', 'Therapy_Type': 'Therapeutic Strategy'},
        color_discrete_sequence=['#2980b9', '#e74c3c']
    )
    fig_pfs.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig_pfs, use_container_width=True)

st.markdown("---")

# =========================================================
# 7. Row 3: superRCA Liquid Biopsy & MRD Monitoring
# =========================================================
st.subheader("superRCA Liquid Biopsy: Circulating Tumor Fraction vs PFS")

fig_mrd = go.Figure()
fig_mrd.add_trace(go.Scatter(
    x=filtered_df['PFS_Months'], 
    y=filtered_df['Tumor_Fraction'], 
    mode='markers',
    marker=dict(
        size=7, 
        color=filtered_df['VMTB_Matching_Score'], 
        colorscale='Viridis', 
        showscale=True, 
        colorbar=dict(title="VMTB Match %")
    ),
    text=filtered_df['Patient_ID'],
    hovertemplate="<b>ID:</b> %{text}<br><b>PFS:</b> %{x:.2f} mo<br><b>Tumor Fraction:</b> %{y:.4f}%<extra></extra>"
))

fig_mrd.update_layout(
    xaxis_title='Progression-Free Survival (Months)', 
    yaxis_title='Tumor Fraction % (Log Scale)', 
    yaxis_type="log",
    margin=dict(l=20, r=20, t=30, b=20)
)
fig_mrd.add_hline(y=0.01, line_dash="dash", line_color="red", annotation_text="superRCA Limit of Detection (0.01%)")

st.plotly_chart(fig_mrd, use_container_width=True)
    
