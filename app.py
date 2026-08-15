import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration & Layout
st.set_page_config(page_title="TCF-001 TRACK Trial Dashboard", layout="wide")
st.title("TCF-001 TRACK Trial: Clinical Outcomes Dashboard")
st.markdown("### Precision Oncology Analytics: VMTB Matching, Survival, and Diagnostic Efficacy")

# 2. Unified Synthetic Data Generation
@st.cache_data
def generate_combined_clinical_data(n_patients=500):
    np.random.seed(42)
    # Merged cohort list from both scripts
    cancer_types = ['Cholangiocarcinoma', 'Primary Uveal Melanoma', 'Chordoma', 'Sarcoma', 'Refractory General']
    
    data = pd.DataFrame({
        'Patient_ID': [f"PT-{i:04d}" for i in range(1, n_patients + 1)],
        'Cohort': np.random.choice(cancer_types, n_patients, p=[0.25, 0.15, 0.15, 0.15, 0.30]),
        'KRAS_Mutant': np.random.choice(['Yes', 'No'], n_patients, p=[0.20, 0.80]),
        'TP53_Mutant': np.random.choice(['Yes', 'No'], n_patients, p=[0.35, 0.65]),
        'VMTB_Matching_Score': np.random.uniform(10, 95, n_patients),
        'Therapy_Type': np.random.choice(['Targeted/Matched', 'Standard Care'], n_patients, p=[0.5, 0.5])
    })
    
    # Biostatistical bias logic (Merged)
    base_pfs = np.random.exponential(scale=8.0, size=n_patients)
    
    # Adjust scores and PFS based on therapy and cohort
    targeted_mask = data['Therapy_Type'] == 'Targeted/Matched'
    
    # General therapy boost
    data.loc[targeted_mask, 'VMTB_Matching_Score'] = np.random.uniform(50, 99, sum(targeted_mask))
    therapy_multiplier = np.where(targeted_mask, 1.8, 1.0)
    
    # KRAS penalty
    kras_penalty = np.where(data['KRAS_Mutant'] == 'Yes', 0.7, 1.0)
    
    # Final PFS Calculation
    data['PFS_Months'] = base_pfs * therapy_multiplier * kras_penalty
    
    # specific Cohort boosts (from App 2)
    um_targeted = (data['Cohort'] == 'Primary Uveal Melanoma') & targeted_mask
    data.loc[um_targeted, 'PFS_Months'] = np.random.uniform(18, 48, sum(um_targeted))
    
    ch_targeted = (data['Cohort'] == 'Chordoma') & targeted_mask
    data.loc[ch_targeted, 'PFS_Months'] = np.random.uniform(12, 36, sum(ch_targeted))
    
    # Progression Event generation (1 = progressed/dead, 0 = censored)
    data['Progression_Event'] = np.random.choice([1, 0], n_patients, p=[0.75, 0.25]) 
    
    # Tumor Fraction for MRD detection
    tumor_fraction = np.random.lognormal(mean=-5, sigma=2, size=n_patients)
    data['Tumor_Fraction'] = np.clip(tumor_fraction, 0.0001, 10)

    return data

df = generate_combined_clinical_data()

# 3. Sidebar Global Filters
st.sidebar.header("Clinical Filters")
selected_cohort = st.sidebar.multiselect("Select Patient Cohort", df['Cohort'].unique(), default=df['Cohort'].unique())
kras_status = st.sidebar.radio("KRAS Mutation Status", ['All', 'Yes', 'No'])

# Apply filters
filtered_df = df[df['Cohort'].isin(selected_cohort)]
if kras_status != 'All':
    filtered_df = filtered_df[filtered_df['KRAS_Mutant'] == kras_status]

# 4. Top-Level Metrics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Evaluable Patients", len(filtered_df))
col2.metric("Median PFS (Months)", round(filtered_df['PFS_Months'].median(), 1))
matched_rate = round((len(filtered_df[filtered_df['Therapy_Type'] == 'Targeted/Matched']) / len(filtered_df)) * 100, 1) if len(filtered_df) > 0 else 0
col3.metric("Matched Therapy Rate", f"{matched_rate}%")
col4.metric("Mean VMTB Score", round(filtered_df['VMTB_Matching_Score'].mean(), 1))
st.markdown("---")

# 5. Visualizations - Row 1: Genomics and Survival (Matplotlib/Seaborn)
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.subheader("Genomic Variant Distribution")
    variant_counts = {
        'TP53': len(filtered_df[filtered_df['TP53_Mutant'] == 'Yes']),
        'KRAS': len(filtered_df[filtered_df['KRAS_Mutant'] == 'Yes']),
        'CDKN2A': int(len(filtered_df) * 0.29),
        'MTAP': int(len(filtered_df) * 0.17)
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=list(variant_counts.keys()), y=list(variant_counts.values()), palette="viridis", ax=ax)
    ax.set_ylabel("Number of Patients")
    ax.set_title("Pathogenic Alterations by Frequency")
    st.pyplot(fig)

with col_viz2:
    st.subheader("Kaplan-Meier Survival Analysis")
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    
    matched = filtered_df[filtered_df['Therapy_Type'] == 'Targeted/Matched']
    unmatched = filtered_df[filtered_df['Therapy_Type'] == 'Standard Care']
    
    if not matched.empty and not unmatched.empty:
        kmf.fit(durations=matched['PFS_Months'], event_observed=matched['Progression_Event'], label='Targeted/Matched')
        kmf.plot_survival_function(ax=ax, color='#3498db', ci_show=True)
        
        kmf.fit(durations=unmatched['PFS_Months'], event_observed=unmatched['Progression_Event'], label='Standard Care')
        kmf.plot_survival_function(ax=ax, color='#e74c3c', ci_show=True)
        
        results = logrank_test(matched['PFS_Months'], unmatched['PFS_Months'], 
                               event_observed_A=matched['Progression_Event'], event_observed_B=unmatched['Progression_Event'])
        
        ax.set_xlabel("Timeline (Months)")
        ax.set_ylabel("Probability of Survival $S(t)$")
        ax.text(0.6, 0.8, f"Log-rank p: {results.p_value:.4f}", transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        st.pyplot(fig)
    else:
        st.warning("Insufficient data in selected cohorts to compute survival curves.")

st.markdown("---")

# 6. Visualizations - Row 2: Matching Scores and Therapy Efficacy (Plotly)
col_viz3, col_viz4 = st.columns(2)

with col_viz3:
    st.subheader("VMTB Matching Scores")
    fig_matching = px.histogram(
        filtered_df, x='VMTB_Matching_Score', color='Therapy_Type', nbins=20,
        labels={'VMTB_Matching_Score': 'Matching Score (%)'},
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    fig_matching.add_vline(x=50, line_dash="dash", line_color="green", annotation_text=">= 50% Actionable")
    fig_matching.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_matching, use_container_width=True)

with col_viz4:
    st.subheader("PFS by Therapy Type")
    fig_pfs = px.box(
        filtered_df, x='Therapy_Type', y='PFS_Months', color='Therapy_Type',
        labels={'PFS_Months': 'PFS (Months)'},
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    fig_pfs.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig_pfs, use_container_width=True)

st.markdown("---")

# 7. Visualizations - Row 3: Liquid Biopsy MRD Detection (Plotly)
st.subheader("superRCA Liquid Biopsy: MRD Detection Sensitivity")

fig_mrd = go.Figure()
fig_mrd.add_trace(go.Scatter(
    x=filtered_df['PFS_Months'], 
    y=filtered_df['Tumor_Fraction'], 
    mode='markers',
    marker=dict(
        size=8, 
        color=filtered_df['VMTB_Matching_Score'], 
        colorscale='Viridis', 
        showscale=True, 
        colorbar=dict(title="VMTB Score")
    ),
    text=filtered_df['Patient_ID'],
    hovertemplate="<b>Patient:</b> %{text}<br><b>PFS:</b> %{x:.1f} months<br><b>Tumor Fraction:</b> %{y:.4f}%<extra></extra>"
))

fig_mrd.update_layout(
    xaxis_title='Progression-Free Survival (Months)', 
    yaxis_title='Tumor Fraction % (Log Scale)', 
    yaxis_type="log",
    margin=dict(l=20, r=20, t=30, b=20)
)
fig_mrd.add_hline(y=0.01, line_dash="dash", line_color="red", annotation_text="superRCA <0.01% Threshold")

st.plotly_chart(fig_mrd, use_container_width=True)
    
