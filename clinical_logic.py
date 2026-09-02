# clinical_logic.py

def assign_escat_tier(row):
    """Dynamically determines the ESCAT Actionability Tier."""
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
    """
    Computes S_VMTB = (E_tier * M_tx * Omega_epistasis) * C_ctDNA * Phi_host
    """
    # 1. Base ESCAT Tier (E_tier)
    tier = row.get('ESCAT_Tier', 'Tier IV')
    if tier == 'Tier I': e_tier = 100.0
    elif tier == 'Tier II': e_tier = 75.0
    elif tier == 'Tier III': e_tier = 50.0
    else: e_tier = 25.0 
        
    # 2. superRCA Limit of Detection (C_ctDNA)
    try:
        tumor_fraction = float(row.get('Tumor_Fraction', 0.05))
    except (ValueError, TypeError):
        tumor_fraction = 0.05
    c_ctdna = 1.0 if tumor_fraction >= 0.01 else 0.70
    
    # 3. Therapeutic Match (M_tx)
    therapy = str(row.get('Therapy_Type', 'Standard Care')).strip()
    m_tx = 1.0 if therapy == 'Targeted/Matched' else 0.30
    
    # 4. Epistatic Penalty (Omega_epistasis)
    tp53 = str(row.get('TP53_Mutant', 'No')).strip().title() == 'Yes'
    omega_epistasis = 0.75 if tp53 else 1.0
    
    # 5. Host-Tumor Metabolic Modulation (Phi_host)
    phi_host = 1.0
    try:
        sii = float(row.get('SII', 500)) 
        ast = float(row.get('AST', 25))
        alt = float(row.get('ALT', 25))
        de_ritis = ast / alt if alt > 0 else 1.0
        
        # Penalize for severe systemic inflammation or subclinical liver dysfunction (MASLD proxy)
        if sii > 800: phi_host -= 0.10
        if de_ritis > 1.2: phi_host -= 0.10
    except (ValueError, TypeError):
        pass

    return min(100.0, round(e_tier * c_ctdna * m_tx * omega_epistasis * phi_host, 1))
    
