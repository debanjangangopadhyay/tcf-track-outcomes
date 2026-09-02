import pytest
from clinical_logic import assign_escat_tier, calculate_vmtb_score

# ==========================================
# 1. ESCAT Tier Mapping Tests
# ==========================================

def test_escat_tier_lung_kras():
    """Test that KRAS in Lung Adenocarcinoma maps to FDA-approved Tier I."""
    row = {'KRAS_Mutant': 'Yes', 'TP53_Mutant': 'No', 'Cohort': 'LUAD'}
    assert assign_escat_tier(row) == 'Tier I'

def test_escat_tier_pancreatic_kras():
    """Test that KRAS in Pancreatic cancer maps to Investigational Tier II."""
    row = {'KRAS_Mutant': 'Yes', 'TP53_Mutant': 'No', 'Cohort': 'PAAD'}
    assert assign_escat_tier(row) == 'Tier II'

def test_escat_tier_tp53_only():
    """Test that TP53 without actionable targets defaults to Tier IV."""
    row = {'KRAS_Mutant': 'No', 'TP53_Mutant': 'Yes', 'Cohort': 'BRCA'}
    assert assign_escat_tier(row) == 'Tier IV'

# ==========================================
# 2. VMTB Core Formula Tests
# ==========================================

def test_vmtb_perfect_match():
    """Test maximum score: Tier I, High ctDNA, Targeted Tx, No Resistance/Stress."""
    row = {
        'ESCAT_Tier': 'Tier I', 
        'Tumor_Fraction': 0.05, 
        'Therapy_Type': 'Targeted/Matched',
        'TP53_Mutant': 'No',
        'AST': 25, 'ALT': 25, 'SII': 500  # Normal metabolic baselines
    }
    # 100 * 1.0 (ctDNA) * 1.0 (Tx) * 1.0 (Epistasis) * 1.0 (Host) = 100.0
    assert calculate_vmtb_score(row) == 100.0

def test_vmtb_subclonal_penalty():
    """Test penalty when ctDNA is below the 0.01% Limit of Detection."""
    row = {
        'ESCAT_Tier': 'Tier I', 
        'Tumor_Fraction': 0.005, 
        'Therapy_Type': 'Targeted/Matched',
        'TP53_Mutant': 'No'
    }
    # 100 * 0.70 (ctDNA) * 1.0 * 1.0 * 1.0 = 70.0
    assert calculate_vmtb_score(row) == 70.0

def test_vmtb_unmatched_penalty():
    """Test severe penalty when standard care is administered instead of targeted."""
    row = {
        'ESCAT_Tier': 'Tier I', 
        'Tumor_Fraction': 0.05, 
        'Therapy_Type': 'Standard Care',
        'TP53_Mutant': 'No'
    }
    # 100 * 1.0 * 0.30 (Tx) * 1.0 * 1.0 = 30.0
    assert calculate_vmtb_score(row) == 30.0

# ==========================================
# 3. Integrative Host-Tumor (Epistasis & MASLD) Tests
# ==========================================

def test_epistatic_resistance():
    """Test that a TP53 co-mutation applies the Omega resistance penalty."""
    row = {
        'ESCAT_Tier': 'Tier I', 
        'Tumor_Fraction': 0.05, 
        'Therapy_Type': 'Targeted/Matched',
        'TP53_Mutant': 'Yes',  # Omega = 0.75
        'AST': 25, 'ALT': 25, 'SII': 500
    }
    # 100 * 1.0 * 1.0 * 0.75 * 1.0 = 75.0
    assert calculate_vmtb_score(row) == 75.0

def test_masld_penalty():
    """Test that De Ritis Ratio > 1.2 triggers a 10% Phi_host penalty."""
    row = {
        'ESCAT_Tier': 'Tier I', 
        'Therapy_Type': 'Targeted/Matched', 
        'Tumor_Fraction': 0.05,
        'TP53_Mutant': 'No', 
        'AST': 40, 'ALT': 25,  # De Ritis = 1.6
        'SII': 500 
    }
    # 100 * 1.0 * 1.0 * 1.0 * 0.90 = 90.0
    assert calculate_vmtb_score(row) == 90.0

def test_heavy_metabolic_stress():
    """Verify Epistasis + MASLD + Systemic Inflammation penalties stack correctly."""
    row = {
        'ESCAT_Tier': 'Tier I', 
        'Therapy_Type': 'Targeted/Matched', 
        'Tumor_Fraction': 0.05,
        'TP53_Mutant': 'Yes',  # Omega = 0.75
        'AST': 40, 'ALT': 25,  # De Ritis > 1.2 (-0.10)
        'SII': 900             # SII > 800 (-0.10)
    }
    # Phi_host = 1.0 - 0.10 - 0.10 = 0.80
    # 100 * 1.0 * 1.0 * 0.75 * 0.80 = 60.0
    assert calculate_vmtb_score(row) == 60.0
    
