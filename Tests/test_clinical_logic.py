import pytest
import pandas as pd
import numpy as np

# Replicate your core functions here for testing
def assign_escat_tier(row):
    kras = str(row.get('KRAS_Mutant', 'No')).strip().title() == 'Yes'
    tp53 = str(row.get('TP53_Mutant', 'No')).strip().title() == 'Yes'
    cohort = str(row.get('Cohort', '')).strip().upper()
    
    if kras:
        if cohort in ['LUAD', 'LUSC', 'COAD', 'READ']: return 'Tier I'
        elif cohort in ['PAAD', 'CHOL']: return 'Tier II'
        else: return 'Tier III'
    if tp53:
        return 'Tier IV'
    return 'Tier IV'

def calculate_vmtb_score(row):
    tier = row.get('ESCAT_Tier', 'Tier IV')
    if tier == 'Tier I': e_tier = 100.0
    elif tier == 'Tier II': e_tier = 75.0
    elif tier == 'Tier III': e_tier = 50.0
    else: e_tier = 25.0 
        
    tumor_fraction = float(row.get('Tumor_Fraction', 0.05))
    c_ctdna = 1.0 if tumor_fraction >= 0.01 else 0.70
    
    therapy = str(row.get('Therapy_Type', 'Standard Care')).strip()
    m_tx = 1.0 if therapy == 'Targeted/Matched' else 0.30
    
    return min(100.0, round(e_tier * c_ctdna * m_tx, 1))

# --- Pytest Assertions ---

def test_escat_tier_lung_kras():
    """Test that KRAS in Lung Adenocarcinoma correctly maps to Tier I."""
    row = {'KRAS_Mutant': 'Yes', 'TP53_Mutant': 'No', 'Cohort': 'LUAD'}
    assert assign_escat_tier(row) == 'Tier I'

def test_escat_tier_pancreatic_kras():
    """Test that KRAS in Pancreatic cancer correctly maps to Tier II."""
    row = {'KRAS_Mutant': 'Yes', 'TP53_Mutant': 'No', 'Cohort': 'PAAD'}
    assert assign_escat_tier(row) == 'Tier II'

def test_vmtb_perfect_match():
    """Test maximum score (Tier I, High ctDNA, Targeted Therapy)."""
    row = {'ESCAT_Tier': 'Tier I', 'Tumor_Fraction': 0.05, 'Therapy_Type': 'Targeted/Matched'}
    assert calculate_vmtb_score(row) == 100.0

def test_vmtb_subclonal_penalty():
    """Test penalty when ctDNA is below Limit of Detection (<0.01%)."""
    row = {'ESCAT_Tier': 'Tier I', 'Tumor_Fraction': 0.005, 'Therapy_Type': 'Targeted/Matched'}
    # 100.0 * 0.70 * 1.0 = 70.0
    assert calculate_vmtb_score(row) == 70.0

def test_vmtb_unmatched_penalty():
    """Test severe penalty when standard care is administered instead of targeted."""
    row = {'ESCAT_Tier': 'Tier I', 'Tumor_Fraction': 0.05, 'Therapy_Type': 'Standard Care'}
    # 100.0 * 1.0 * 0.30 = 30.0
    assert calculate_vmtb_score(row) == 30.0
  
