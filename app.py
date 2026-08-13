# Step 1: Install required libraries


# Step 2: Import libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import random  # Required for the dynamic port fix

# ---------------------------------------------------------
# Step 3: Simulate the Clinical Trial Data
# ---------------------------------------------------------

np.random.seed(42)
n_patients = 500

data = {
    'Patient_ID': [f'PT-{i:04d}' for i in range(1, n_patients + 1)],
    'Cohort': np.random.choice(['Refractory General', 'Primary Uveal Melanoma', 'Chordoma'], n_patients, p=[0.6, 0.2, 0.2]),
    'VMTB_Matching_Score': np.random.uniform(10, 95, n_patients),
    'Therapy_Type': np.random.choice(['Standard Care', 'Targeted/Neoadjuvant'], n_patients, p=[0.4, 0.6]),
    'PFS_Months': np.random.uniform(2, 36, n_patients)
}

df = pd.DataFrame(data)

# Adjust data logically based on the abstract's claims
df.loc[df['VMTB_Matching_Score'] >= 50, 'PFS_Months'] *= 1.5

um_targeted = (df['Cohort'] == 'Primary Uveal Melanoma') & (df['Therapy_Type'] == 'Targeted/Neoadjuvant')
df.loc[um_targeted, 'VMTB_Matching_Score'] = np.random.uniform(70, 99, sum(um_targeted))
df.loc[um_targeted, 'PFS_Months'] = np.random.uniform(18, 48, sum(um_targeted))

ch_targeted = (df['Cohort'] == 'Chordoma') & (df['Therapy_Type'] == 'Targeted/Neoadjuvant')
df.loc[ch_targeted, 'VMTB_Matching_Score'] = np.random.uniform(65, 95, sum(ch_targeted))
df.loc[ch_targeted, 'PFS_Months'] = np.random.uniform(12, 36, sum(ch_targeted))

# ---------------------------------------------------------
# Step 4: Build the Standard Dash Application
# ---------------------------------------------------------

app = Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f9f9f9'}, children=[
    
    html.H1("TCF-001 TRACK Trial: Clinical Outcomes Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.P("Precision Oncology Analytics: VMTB Matching, Survival, and Diagnostic Efficacy", style={'textAlign': 'center', 'color': '#7f8c8d'}),
    
    html.Div([
        html.Label("Select Patient Cohort:"),
        dcc.Dropdown(
            id='cohort-filter',
            options=[
                {'label': 'All Participants', 'value': 'All'},
                {'label': 'Primary Uveal Melanoma', 'value': 'Primary Uveal Melanoma'},
                {'label': 'Chordoma', 'value': 'Chordoma'},
                {'label': 'Refractory General', 'value': 'Refractory General'}
            ],
            value='All',
            clearable=False
        )
    ], style={'width': '40%', 'margin': 'auto', 'paddingBottom': '20px'}),

    html.Div([
        html.Div([dcc.Graph(id='matching-score-chart')], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='pfs-chart')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ]),
    
    html.Div([dcc.Graph(id='mrd-detection-chart')], style={'width': '100%', 'marginTop': '20px'})
])

# ---------------------------------------------------------
# Step 5: Define Callbacks
# ---------------------------------------------------------

@app.callback(
    [Output('matching-score-chart', 'figure'),
     Output('pfs-chart', 'figure'),
     Output('mrd-detection-chart', 'figure')],
    [Input('cohort-filter', 'value')]
)
def update_charts(selected_cohort):
    
    if selected_cohort == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['Cohort'] == selected_cohort]
        
    fig_matching = px.histogram(
        filtered_df, x='VMTB_Matching_Score', color='Therapy_Type', nbins=20,
        title=f'VMTB Matching Scores ({selected_cohort})',
        labels={'VMTB_Matching_Score': 'Matching Score (%)'},
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    fig_matching.add_vline(x=50, line_dash="dash", line_color="green", annotation_text=">= 50% Actionable Threshold")
    
    fig_pfs = px.box(
        filtered_df, x='Therapy_Type', y='PFS_Months', color='Therapy_Type',
        title=f'Progression-Free Survival by Therapy ({selected_cohort})',
        labels={'PFS_Months': 'PFS (Months)'},
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    
    tumor_fraction = np.random.lognormal(mean=-5, sigma=2, size=len(filtered_df))
    tumor_fraction = np.clip(tumor_fraction, 0.0001, 10)
    
    fig_mrd = go.Figure()
    fig_mrd.add_trace(go.Scatter(
        x=filtered_df['PFS_Months'], y=tumor_fraction, mode='markers',
        marker=dict(size=8, color=filtered_df['VMTB_Matching_Score'], colorscale='Viridis', showscale=True, colorbar=dict(title="VMTB Score")),
        text=filtered_df['Patient_ID']
    ))
    
    fig_mrd.update_layout(
        title='superRCA Liquid Biopsy: MRD Detection Sensitivity',
        xaxis_title='Progression-Free Survival (Months)', yaxis_title='Tumor Fraction % (Log Scale)', yaxis_type="log"
    )
    fig_mrd.add_hline(y=0.01, line_dash="dash", line_color="red", annotation_text="superRCA <0.01% Threshold")

    return fig_matching, fig_pfs, fig_mrd

# ---------------------------------------------------------
# Step 6: Run the App with a Dynamic Random Port
# ---------------------------------------------------------
if __name__ == '__main__':
    # Generates a random port between 8050 and 9050 every time the cell runs.
    # This prevents Colab from crashing due to a cached/locked background server.
    dynamic_port = random.randint(8050, 9050)
    app.run(jupyter_mode='inline', port=dynamic_port, jupyter_height=850)
