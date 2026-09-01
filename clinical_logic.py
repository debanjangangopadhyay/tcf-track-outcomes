# clinical_logic.py

def assign_escat_tier(row):
    """
    Dynamically determines the ESCAT Actionability Tier based on TCGA Cohort and Mutations.
    """
    kras = str(row.get('KRAS_Mutant', 'No')).strip().title() == 'Yes'
    tp53 = str(row.get('TP53_Mutant', 'No')).strip().title() == 'Yes'
    cohort = str(row.get('Cohort', '')).strip().upper()
    
    # 1. Evaluate KRAS Actionability
    if kras:
        if cohort in ['LUAD', 'LUSC', 'COAD', 'READ']: return 'Tier I'
        elif cohort in ['PAAD', 'CHOL']: return 'Tier II'
        else: return 'Tier III'
        
    # 2. Evaluate TP53 Actionability
    if tp53:
        return 'Tier IV'
        
    # 3. Default for Wildtype or unmapped variants
    return 'Tier IV'

def calculate_vmtb_score(row):
    """
    Deterministic VMTB matching score based on computed ESCAT tiers, 
    superRCA ctDNA confidence, and therapeutic administration.
    """
    tier = row.get('ESCAT_Tier', 'Tier IV')
    if tier == 'Tier I': e_tier = 100.0
    elif tier == 'Tier II': e_tier = 75.0
    elif tier == 'Tier III': e_tier = 50.0
    else: e_tier = 25.0 
        
    # Limit of Detection for superRCA is 0.01%
    tumor_fraction = float(row.get('Tumor_Fraction', 0.05))
    c_ctdna = 1.0 if tumor_fraction >= 0.01 else 0.70
    
    therapy = str(row.get('Therapy_Type', 'Standard Care')).strip()
    m_tx = 1.0 if therapy == 'Targeted/Matched' else 0.30
    
    return min(100.0, round(e_tier * c_ctdna * m_tx, 1))
  
