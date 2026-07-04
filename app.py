import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="HDI Predictor Pro",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design and crisp typography
st.markdown("""
<style>
    /* Main body background tweak */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Result Box Styling with white text for high contrast */
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .result-card h2 {
        margin: 0 !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
    }
    .result-card p {
        margin-top: 10px !important;
        font-size: 1.1rem !important;
        opacity: 0.9;
    }
    
    /* Dynamic color classes with crisp white/light-gray text */
    .very-high-card { background: linear-gradient(135deg, #1e4620, #0f2310); border-left: 6px solid #28a745; color: #ffffff !important; }
    .very-high-card h2 { color: #28a745 !important; }
    
    .high-card { background: linear-gradient(135deg, #1b3a4b, #0d1d26); border-left: 6px solid #007bff; color: #ffffff !important; }
    .high-card h2 { color: #007bff !important; }
    
    .medium-card { background: linear-gradient(135deg, #4d3d0c, #261f06); border-left: 6px solid #ffc107; color: #ffffff !important; }
    .medium-card h2 { color: #ffc107 !important; }
    
    .low-card { background: linear-gradient(135deg, #4d1c1c, #260e0e); border-left: 6px solid #dc3545; color: #ffffff !important; }
    .low-card h2 { color: #dc3545 !important; }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('hdi_model.joblib')
        scaler = joblib.load('hdi_scaler.joblib')
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

# App Header
st.title("🌍 Human Development Index Predictor")
st.caption("Analyze and predict a country's HDI tier based on global health, education, and economic indicators.")
st.write("---")

if model is None:
    st.error("⚠️ **Model files missing.** Please run `python train_model.py` first to train and serialize your model (`hdi_model.joblib` and `hdi_scaler.joblib`).")
    st.stop()

# Sidebar Setup
st.sidebar.header("📊 Input Parameters")
st.sidebar.markdown("Fine-tune the individual dimensions or select a standard preset below.")

# Scenario presets
scenario = st.sidebar.selectbox(
    "⚡ Quick Preset Scenarios:",
    ["Custom", "Very High Development", "High Development", "Medium Development", "Low Development"]
)

# Set defaults based on preset selections
defaults = {
    "Custom": (75.0, 8.0, 13.0, 15000),
    "Very High Development": (82.0, 12.5, 17.0, 45000),
    "High Development": (75.0, 9.5, 14.0, 18000),
    "Medium Development": (68.0, 7.0, 12.0, 8000),
    "Low Development": (58.0, 4.0, 9.0, 2000)
}

le_d, ms_d, es_d, gni_d = defaults[scenario]

# Input sliders (locked to preset values if a preset is selected)
life_expectancy = st.sidebar.slider(
    "🏥 Life Expectancy (years)", 40.0, 90.0, le_d, 0.5,
    help="Average lifespan at birth. Reflects systemic healthcare and living conditions."
)

mean_years_schooling = st.sidebar.slider(
    "🎓 Mean Years of Schooling", 0.0, 15.0, ms_d, 0.5,
    help="Average years of education completed by adults aged 25+."
)

expected_years_schooling = st.sidebar.slider(
    "📚 Expected Years of Schooling", 5.0, 22.0, es_d, 0.5,
    help="Total anticipated years of schooling for an entering child."
)

gni_per_capita = st.sidebar.slider(
    "💰 GNI per Capita (PPP $)", 500, 80000, gni_d, 500,
    help="Gross National Income converted to international dollars using purchasing power parity rates."
)

# Layout Setup: 2 columns for a balanced visual workspace
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📋 Input Configurations")
    
    # Using streamlined native columns for clean, card-like metrics
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="Life Expectancy", value=f"{life_expectancy:.1f} Yrs")
        st.metric(label="Expected Schooling", value=f"{expected_years_schooling:.1f} Yrs")
    with m_col2:
        st.metric(label="Mean Schooling", value=f"{mean_years_schooling:.1f} Yrs")
        st.metric(label="GNI per Capita", value=f"${gni_per_capita:,.0f}")
        
    st.write("")
    predict_btn = st.button("Generate HDI Prediction", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Result")
    
    # Trigger prediction automatically on preset changes or on explicit click
    if predict_btn or scenario != "Custom":
        # Feature vector assembly
        input_features = np.array([[life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita]])
        input_scaled = scaler.transform(input_features)
        
        # Pipeline execution
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        prob_dict = dict(zip(model.classes_, probabilities))
        
        # UI mapping tags
        tier_slugs = {'Very High': 'very-high', 'High': 'high', 'Medium': 'medium', 'Low': 'low'}
        tier_descriptions = {
            'Very High': 'This profile reflects exceptional infrastructure, strong educational pipelines, and superior public health systems.',
            'High': 'This profile demonstrates solid socio-economic frameworks with moderate avenues available for structural improvement.',
            'Medium': 'This profile indicates developing structural milestones. Targeted public investments would yield high societal returns.',
            'Low': 'This profile faces critical systemic headwinds across economic resources, school accessibility, and healthcare.'
        }
        
        # Output card rendering with clean, white contrast texts
        st.markdown(f'''
        <div class="result-card {tier_slugs[prediction]}-card">
            <h2>{prediction} HDI</h2>
            <p>{tier_descriptions[prediction]}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Probability Chart Configuration
        fig = px.bar(
            x=list(prob_dict.values()),
            y=list(prob_dict.keys()),
            orientation='h',
            labels={'x': 'Confidence Score', 'y': 'HDI Classification'},
            title="Classification Confidence Spread",
            color=list(prob_dict.keys()),
            color_discrete_map={
                'Very High': '#28a745',
                'High': '#007bff',
                'Medium': '#ffc107',
                'Low': '#dc3545'
            }
        )
        fig.update_layout(
            showlegend=False, 
            height=240, 
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_tickformat='.0%'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("💡 Adjust the sidebar features and click 'Generate HDI Prediction' to see results.")

# Metadata Breakdown Layout 
st.write("---")
expander = st.開設_expander if hasattr(st, "開設_expander") else st.expander("📖 View HDI Classification Index Metrics")
with expander:
    tier_info = pd.DataFrame({
        'Tier Classification': ['Very High Development', 'High Development', 'Medium Development', 'Low Development'],
        'Official HDI Range': ['≥ 0.800', '0.700 - 0.799', '0.550 - 0.699', '< 0.550'],
        'Structural Benchmarks': [
            'Fully realized public healthcare, mandatory access to tertiary schooling, strong economic capacity.',
            'Stable industrial capabilities, rising secondary education attainment rates, developing service sectors.',
            'Emerging market transitions, infrastructure variations, expanding foundational primary access.',
            'Resource constrained settings, ongoing infrastructure programs, active development target profiles.'
        ]
    })
    st.table(tier_info)