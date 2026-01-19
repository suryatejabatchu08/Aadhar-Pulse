# Aadhaar Pulse - UIDAI Hackathon 2026 Project

**Aadhaar Pulse** is a privacy-preserving, analytics-driven solution designed to identify societal trends, operational risks, and predictive indicators using **only official UIDAI aggregated datasets**.

## 🚀 Key Features
1.  **Aadhaar Stress Index**: A composite score (0-100) determining the operational pressure on districts based on Enrolment, Biometric, and Demographic update volumes.
2.  **Predictive Modeling**: Forecasts future demand for updates using time-series analysis (Prophet).
3.  **Actionable Recommendations**: Auto-generated policy actions (e.g., "Deploy Mobile Camps") based on specific stress factors.
4.  **Interactive Dashboard**: A clean Streamlit UI for policymakers to visualize trends and hotspots.

## 📂 Data & Privacy
- **Source**: Aggregated UIDAI datasets (Enrolment, Demographic Updates, Biometric Updates).
- **Privacy**: No PII (Personally Identifiable Information) is used. All analysis is performed on aggregated counts (State/District/Age-Group level).
- **Constraint Compliance**: Zero external data used. Geo-spatial analysis relies on inherent State/District hierarchies.

## 🛠️ Technical Architecture
- **Data Ingestion**: `src/data_loader.py` consolidates multi-part CSVs and aggregates daily logs into monthly trends.
- **Logic Core**:
    - `src/stress_index.py`: Calculates normalized stress scores.
    - `src/models.py`: Runs anomaly detection (Z-score) and forecasting (Prophet).
    - `src/recommender.py`: Rule-based engine for insights.
- **UI**: Streamlit dashboard (`src/dashboard.py`).

## ⚙️ Setup & Usage

### Prerequisites
- Python 3.8+
- Libraries: `pandas`, `numpy`, `streamlit`, `plotly`, `prophet`, `statsmodels`

### Installation
```bash
pip install -r requirements.txt
```

### Running the System
1.  **Process Data**:
    ```bash
    python src/data_loader.py
    python src/stress_index.py
    python src/models.py
    python src/recommender.py
    ```
2.  **Launch Dashboard**:
    ```bash
    streamlit run src/dashboard.py
    ```

## 📊 Assumptions
- **Age as Proxy**: Since specific update types (e.g., "Iris" vs "Fingerprint") were not explicitly separated in the provided schema but Age Groups were (0-5, 5-17, 18+), we use Age Groups as proxies:
    - **5-17**: High correlation with Mandatory Biometric Updates.
    - **0-5**: High correlation with New Enrolments.
    - **18+**: High correlation with Demographic Corrections or Voluntary Updates.
- **Geography**: Analysis is performed at the District level. Pin codes are aggregated up to Districts where possible or treated mainly within the raw data context.

---
*Built for UIDAI Hackathon 2026*
