# tcf-track-outcomes
# Decentralized Genomic Profiling & Clinical Outcomes Dashboard

A full-stack clinical decision support and biostatistical web application engineered for precision oncology analytics. This tool bridges the gap between raw clinical-genomic data and actionable survival outcomes, providing rapid insights into therapeutic matching and disease progression.

🔗 **Live Application:** https://tcf-track-outcomes.onrender.com 

---

## 📊 Key Features
* **Kaplan-Meier Survival Analysis:** Programmatically computes progression-free survival (PFS) curves and calculates log-rank statistical significance ($p$-values) comparing targeted versus standard care regimens.
* **Virtual Molecular Tumor Board (VMTB):** Visualizes genomic actionability matching scores across cohorts.
* **superRCA Liquid Biopsy Integration:** Evaluates circulating tumor fractions against longitudinal progression tracking.
* **Multi-Cohort Scalability:** Out-of-the-box support for over 30 cancer cohorts derived from the TCGA Pan-Cancer Clinical Data Resource, with specialized focus on hepatobiliary and metabolic disease pathways.

---

## 🛠️ Tech Stack
* **Frontend/UI:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Survival Biostatistics:** Lifelines, SciPy
* **Visualizations:** Plotly, Seaborn, Matplotlib

---

## ⚙️ Local Installation & Running

1. Clone the repository:
   ```bash
   git clone [https://github.com/debanjangangopadhyay/tcf-track-outcomes.git](https://github.com/debanjangangopadhyay/tcf-track-outcomes.git)
   cd tcf-track-outcomes
   
