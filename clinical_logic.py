# clinical_logic.py
import math

def assign_escat_tier(row):
    """Dynamically determines the ESCAT Actionability Tier."""
    kras = str(row.get('KRAS_Mutant', 'No')).strip().title() == 'Yes'
    tp53 = str(row.get('TP53_Mutant', 'No')).strip().title() == 'Yes'
    cohort = str(row.get('Cohort', '')).strip().upper()
    
    if kras:
        if cohort in ['LUAD', 'LUSC', 'COAD', 'READ']: return 'Tier I'
        elif cohort in ['PAAD', 'CHOL']: return 'Tier II'
        else: return 'Tier III'
        
    if tp53: return 'Tier IV'
    return 'Tier IV'

def calculate_vmtb_score(row):
    """
    Computes the advanced Clinical Utility Index (CUI).
    Integrates ESCAT, Epistasis, MASLD proxy, Toxicity, and Clonal Kinetics.
    """
    # 1. Base ESCAT Tier (E_tier)
    tier = row.get('ESCAT_Tier', 'Tier IV')
    e_tier = {'Tier I': 100.0, 'Tier II': 75.0, 'Tier III': 50.0}.get(tier, 25.0)
        
    # 2. Therapeutic Match (M_tx)
    therapy = str(row.get('Therapy_Type', 'Standard Care')).strip()
    m_tx = 1.0 if therapy == 'Targeted/Matched' else 0.30
    
    # 3. Epistatic Penalty (Omega_epistasis)
    tp53 = str(row.get('TP53_Mutant', 'No')).strip().title() == 'Yes'
    omega_epistasis = 0.75 if tp53 else 1.0
    
    # 4. Host-Tumor Metabolic Modulation (Phi_host) & Toxicity (Tau_tox)
    phi_host = 1.0
    tau_tox = 0.0
    try:
        sii = float(row.get('SII', 500)) 
        ast = float(row.get('AST', 25))
        alt = float(row.get('ALT', 25))
        age = float(row.get('Age', 50))
        
        de_ritis = ast / alt if alt > 0 else 1.0
        
        # Phi_host: Microenvironment hostility
        if sii > 800: phi_host -= 0.10
        if de_ritis > 1.2: phi_host -= 0.10
        
        # Tau_tox: Pharmacometabolic clearance risks
        if age > 65: tau_tox += 0.05
        if de_ritis > 1.2: tau_tox += 0.15
        if de_ritis > 2.0: tau_tox += 0.15 # Severe hepatotoxicity risk
        
    except (ValueError, TypeError):
        pass

    # 5. Longitudinal Clonal Clearance Kinetics (K_clearance)
    k_clearance = 1.0
    try:
        f_base = float(row.get('Tumor_Fraction', 0.05))
        f_follow = row.get('Tumor_Fraction_Followup', None)
        
        if f_follow is not None and not math.isnan(float(f_follow)):
            f_follow = float(f_follow)
            if f_base > 0 and f_follow > 0:
                v_clearance = math.log(f_base / f_follow)
                
                if v_clearance <= 0:
                    k_clearance = 0.50  # Molecular progression (tumor growing)
                elif v_clearance < 0.693:
                    k_clearance = 0.80  # Slow clearance (less than 50% drop)
                # else remains 1.0 for optimal clearance
    except (ValueError, TypeError):
        pass
    
    # 6. Calculate Final Clinical Utility Index (CUI)
    base_vmtb = e_tier * m_tx * omega_epistasis * phi_host
    toxicity_modifier = math.exp(-tau_tox)
    
    cui = base_vmtb * k_clearance * toxicity_modifier
    
    return min(100.0, round(cui, 1))
    
