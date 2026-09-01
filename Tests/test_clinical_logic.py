import pytest
from clinical_logic import assign_escat_tier, calculate_vmtb_score

def test_escat_tier_lung_kras():
    """Test that KRAS in Lung Adenocarcinoma correctly maps to Tier I."""
    row = {'KRAS_Mutant': 'Yes', 'TP53_Mutant': 'No', 'Cohort': 'LUAD'}
    assert assign_escat_tier(row) == 'Tier I'

def test_escat_tier_pancreatic_kras():
    """Test that KRAS in Pancreatic cancer correctly maps to Tier II."""
    row = {'KRAS_Mutant': 'Yes', 'TP53_Mutant': 'No', 'Cohort': 'PAAD'}
    assert assign_escat_tier(row) == 'Tier II'

def test_escat_tier_tp53_only():
    """Test that TP53 without actionable KRAS defaults to Tier IV."""
    row = {'KRAS_Mutant': 'No', 'TP53_Mutant': 'Yes', 'Cohort': 'BRCA'}
    assert assign_escat_tier(row) == 'Tier IV'

def test_vmtb_perfect_match():
    """Test maximum score (Tier I, High ctDNA, Targeted Therapy)."""
    row = {'ESCAT_Tier': 'Tier I', 'Tumor_Fraction': 0.05, 'Therapy_Type': 'Targeted/Matched'}
    assert calculate_vmtb_score(row) == 100.0

def test_vmtb_subclonal_penalty():
    """Test penalty when ctDNA is below Limit of Detection (<0.01%)."""
    row = {'ESCAT_Tier': 'Tier I', 'Tumor_Fraction': 0.005, 'Therapy_Type': 'Targeted/Matched'}
    assert calculate_vmtb_score(row) == 70.0

def test_vmtb_unmatched_penalty():
    """Test severe penalty when standard care is administered instead of targeted."""
    row = {'ESCAT_Tier': 'Tier I', 'Tumor_Fraction': 0.05, 'Therapy_Type': 'Standard Care'}
    assert calculate_vmtb_score(row) == 30.0
    
