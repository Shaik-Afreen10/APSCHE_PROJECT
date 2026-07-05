# 🌍 HDI Predictor Pro: Machine Learning Analytics Dashboard

[![Python Version](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://apscheproject-dgt9nekvubtmtegghanjrf.streamlit.app/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](https://opensource.org/licenses/MIT)

An interactive, end-to-end Machine Learning web application that predicts a country's **Human Development Index (HDI) Classification Tier** using foundational global socio-economic dimensions: Health, Education, and Economic Capacity.

🔗 **Production Deployment:** [Launch Active Dashboard](https://apscheproject-dgt9nekvubtmtegghanjrf.streamlit.app/)

---

## 📋 Table of Contents
* [📌 Core Overview](#-core-overview)
* [✨ Key Features](#-key-features)
* [🧠 Model Architecture & Specification](#-model-architecture--specification)
* [📊 Data Feature Engineering](#-data-feature-engineering)
* [⚙️ Installation & Local Replication](#️-installation--local-replication)
* [📂 Repository Blueprint](#-repository-blueprint)
* [🔄 System Architecture Workflow](#-system-architecture-workflow)
* [🛠️ Technology Stack](#️-technology-stack)
* [📈 Production Roadmap](#-production-roadmap)
* [⚠️ Disclaimer](#️-disclaimer)
* [👩‍💻 Author & Contribution](#-author--contribution)

---

## 📌 Core Overview

The **HDI Predictor Pro** pipeline bridges statistical economic tracking with prescriptive machine learning. By evaluating raw structural inputs, the backend replaces traditional programmatic aggregations with an optimized multi-class classifier.

The system scales across both desktop and mobile environments using responsive container primitives and adaptive theme mapping variables.
[ Socio-Economic Feature Space ]
         (Health, Mean/Expected Education, GNI Log-Scaled)
                               │
                               ▼
                   [ Robust Scaler Pipeline ]
                               │
                               ▼
                 [ Random Forest Classifier ]
                               │
     ┌─────────────────────────┼────────────────────────┐
     ▼                         ▼                        ▼
🟢 Very High HDI            🔵 High HDI             🟡 Medium / 🔴 Low
---

## ✨ Key Features

* **Adaptive UI Framework:** Operates flawlessly across both **Native Light** and **Native Dark** browser presets using design token CSS injections (`var(--text-color)`).
* **Deterministic Inference:** Real-time generation of categorical development bounds instantly powered by vectorized feature matrix evaluation.
* **Granular Confidence Modeling:** Integrates interactive custom `Plotly Go.Indicator` charts alongside relative probability bar matrix breakdowns.
* **Deterministic Configuration States:** Includes zero-latency simulation presets for key developmental tiers to accelerate data profiling tasks.

---

## 🧠 Model Architecture & Specification

The analytical foundation evaluates multi-collinear inputs through an ensemble decision paradigm, neutralizing data sparsity anomalies across boundaries.

| Metric / Parameter | Production Specifications |
| :--- | :--- |
| **Algorithm Base** | `RandomForestClassifier` Ensemble |
| **Estimators ($N$)** | 100 Independent Decision Trees |
| **Max Depth Hyperparameter** | 10 Splits (Optimized against Overfitting) |
| **Vector Dimension** | 4 Dense Numeric Matrix Vectors |
| **Target Vector Classification** | 4 Distinct Stratified Tiers (`Low` ➔ `Very High`) |
| **Inference Serialization** | `Joblib` Compression Protocols |

---

## 📊 Data Feature Engineering

The underlying classification logic evaluates feature arrays along the three official pillars defined by the United Nations Development Programme (UNDP):

### 1. Feature Map Primitives
| Target Input Vector | Unit Domain | Analytical Weight Factor |
| :--- | :--- | :--- |
| **🏥 Life Expectancy at Birth** | Continuous (Years) | Health Dimension Index Proxy |
| **🎓 Mean Years of Schooling** | Continuous (Years) | Adult Educational Attainment Metric |
| **📚 Expected Years of Schooling** | Continuous (Years) | Youth Educational Enrollment Ceiling |
| **💰 GNI per Capita (PPP)** | Constant Int (USD $) | Real Economic Purchasing Power Parity |

### 2. Operational Boundaries (Inference Target Target)
[ Low Development ]    [ Medium Dev ]    [ High Dev ]    [ Very High Development ]
0.000                 0.550             0.700           0.800                     1.000
◄─────────────────────┼─────────────────┼───────────────┼─────────────────────────►
---

## ⚙️ Installation & Local Replication

Follow these step-by-step instructions to set up the runtime environment locally:

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed on your local workspace architecture.

### 2. Clone and Initialize Path
#```bash
# Clone remote repository target
git clone [https://github.com/Shaik-Afreen10/APSCHE_PROJECT.git](https://github.com/Shaik-Afreen10/APSCHE_PROJECT.git)

# Shift operational path context
cd APSCHE_PROJECT

3. Environment Isolation & Package Resolution
Bash
# Initialize clean runtime context environment
python -m venv venv

# Windows Context Activation
venv\Scripts\activate

# macOS / Linux Context Activation
source venv/bin/activate

# Execute core dependancy installation pipeline
pip install -r requirements.txt

4. Execute Model Training & Dashboard Runtime
Bash
# Train, evaluate, and serialize structural components
python train_model.py

# Boot up stream local development instance server
streamlit run app.py

APSCHE_PROJECT/
│
├── app.py                  # Streamlit Web Application & UI Architecture
├── train_model.py          # Scikit-learn Pipeline Training & Evaluation Script
├── requirements.txt        # Deterministic Dependency Matrix Manifest
├── README.md               # Project Documentation & Architecture Blueprint
│
├── hdi_model.joblib        # Serialized Random Forest Model Binaries
├── hdi_scaler.joblib       # Serialized Feature Normalization Transform Scaler
└── hdi_training_data.csv   # Structurally Generated Synthesis Core Dataset

🔄 System Architecture Workflow
User Interaction Layer: User updates variables manually via side panel controls or triggers structured preset profiles.

Pipeline Execution Layer: Raw arrays route directly to the persistent scaler asset file (hdi_scaler.joblib) to adjust value scopes uniformly.

Inference Execution Layer: Scaled inputs route into the ensemble engine (hdi_model.joblib) to extract prediction vectors and classification array states.

Presentation Graph Layer: Renders the dynamic KPI target status boxes, classification gauges, and progressive execution blocks across the page.

🛠️ Technology Stack
Core Language: Python 3.10+

Data Core Engine: Pandas, NumPy

Statistical Inference Engine: Scikit-learn (Random Forest, Preprocessing)

Visual Graph Engine: Plotly Engine (Express & Graph Objects Canvas mapping)

Application Frame Host: Streamlit Web Framework

Object Stream Persistence: Joblib Utility

📈 Production Roadmap
[ ] Transition pipeline tracking to ingest live real-world historical United Nations (UNDP) core data repositories.

[ ] Implement advanced model explanations using SHAP (SHapley Additive exPlanations) frameworks to expose feature weights during live sessions.

[ ] Deploy spatial data integrations using an interactive Folium/Plotly Choropleth World Map Grid.

[ ] Expose an autonomous headless REST API interface endpoint for programmatic third-party querying.
