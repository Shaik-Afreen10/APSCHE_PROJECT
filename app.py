import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="HDI Predictor",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #2d2d2d;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #444;
    }
    .metric-card strong {
        color: #ffffff;
        font-size: 1rem;
    }
    .metric-card span {
        color: #00ff88;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .very-high { background-color: #1a4d2e; border-left: 5px solid #00ff88; }
    .very-high h2 { color: #00ff88; margin: 0; }
    .very-high p { color: #ffffff; margin: 10px 0 0 0; }
    
    .high { background-color: #1a3a5c; border-left: 5px solid #4da6ff; }
    .high h2 { color: #4da6ff; margin: 0; }
    .high p { color: #ffffff; margin: 10px 0 0 0; }
    
    .medium { background-color: #5c4a1a; border-left: 5px solid #ffd700; }
    .medium h2 { color: #ffd700; margin: 0; }
    .medium p { color: #ffffff; margin: 10px 0 0 0; }
    
    .low { background-color: #5c1a1a; border-left: 5px solid #ff4444; }
    .low h2 { color: #ff4444; margin: 0; }
    .low p { color: #ffffff; margin: 10px 0 0 0; }
    
    .stDataFrame {
        color: #ffffff;
    }
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

# Header
st.markdown('<div class="main-header">🌍 Human Development Index Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict HDI tier based on life expectancy, education, and income indicators</div>', unsafe_allow_html=True)

if model is None:
    st.error("Model not found. Please run 'python train_model.py' first to train the model.")
    st.stop()

# Sidebar for input
st.sidebar.header("📊 Input Parameters")
st.sidebar.markdown("**Adjust the sliders below to set country development indicators:**")

# Input sliders
life_expectancy = st.sidebar.slider(
    "🏥 Life Expectancy (years)",
    min_value=40.0,
    max_value=90.0,
    value=75.0,
    step=0.5,
    help="Average number of years a person is expected to live. Higher values indicate better healthcare."
)

mean_years_schooling = st.sidebar.slider(
    "🎓 Mean Years of Schooling",
    min_value=0.0,
    max_value=15.0,
    value=8.0,
    step=0.5,
    help="Average years of education for adults aged 25+. Higher values indicate better education systems."
)

expected_years_schooling = st.sidebar.slider(
    "📚 Expected Years of Schooling",
    min_value=5.0,
    max_value=22.0,
    value=13.0,
    step=0.5,
    help="Years of schooling a child can expect to receive. Higher values indicate better educational opportunities."
)

gni_per_capita = st.sidebar.slider(
    "💰 GNI per Capita (PPP $)",
    min_value=500,
    max_value=80000,
    value=15000,
    step=500,
    help="Gross National Income per person adjusted for cost of living. Higher values indicate stronger economies."
)

# Scenario presets
st.sidebar.subheader("⚡ Quick Scenarios")
st.sidebar.markdown("*Or select a preset scenario to auto-fill values:*")
scenario = st.sidebar.selectbox(
    "Choose scenario:",
    ["Custom", "Very High Development", "High Development", "Medium Development", "Low Development"]
)

if scenario == "Very High Development":
    life_expectancy = 82.0
    mean_years_schooling = 12.5
    expected_years_schooling = 17.0
    gni_per_capita = 45000
elif scenario == "High Development":
    life_expectancy = 75.0
    mean_years_schooling = 9.5
    expected_years_schooling = 14.0
    gni_per_capita = 18000
elif scenario == "Medium Development":
    life_expectancy = 68.0
    mean_years_schooling = 7.0
    expected_years_schooling = 12.0
    gni_per_capita = 8000
elif scenario == "Low Development":
    life_expectancy = 58.0
    mean_years_schooling = 4.0
    expected_years_schooling = 9.0
    gni_per_capita = 2000

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Input Summary")
    
    # Display input values
    input_data = {
        "Life Expectancy": f"{life_expectancy:.1f} years",
        "Mean Years of Schooling": f"{mean_years_schooling:.1f} years",
        "Expected Years of Schooling": f"{expected_years_schooling:.1f} years",
        "GNI per Capita": f"${gni_per_capita:,.0f}"
    }
    
    for key, value in input_data.items():
        st.markdown(f'<div class="metric-card"><strong>{key}:</strong> <span>{value}</span></div>', unsafe_allow_html=True)
    
    # Predict button
    predict_btn = st.button("Predict HDI Tier", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Result")
    
    if predict_btn or scenario != "Custom":
        # Prepare input for prediction
        input_features = np.array([[life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita]])
        input_scaled = scaler.transform(input_features)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get class labels
        classes = model.classes_
        prob_dict = dict(zip(classes, probabilities))
        
        # Display result
        tier_colors = {
            'Very High': 'very-high',
            'High': 'high',
            'Medium': 'medium',
            'Low': 'low'
        }
        
        tier_descriptions = {
            'Very High': 'This country demonstrates exceptional human development with strong performance across all dimensions.',
            'High': 'This country shows good human development with room for improvement in specific areas.',
            'Medium': 'This country has moderate human development and could benefit from targeted investments.',
            'Low': 'This country faces significant development challenges requiring comprehensive intervention.'
        }
        
        st.markdown(f'''
        <div class="result-box {tier_colors[prediction]}">
            <h2 style="margin: 0; color: #333;">{prediction} HDI</h2>
            <p style="margin: 10px 0 0 0; color: #555;">{tier_descriptions[prediction]}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Display probabilities
        st.subheader("📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            'HDI Tier': list(prob_dict.keys()),
            'Probability': [f"{p:.1%}" for p in prob_dict.values()]
        })
        st.table(prob_df)
        
        # Create probability chart
        fig = px.bar(
            x=list(prob_dict.keys()),
            y=list(prob_dict.values()),
            labels={'x': 'HDI Tier', 'y': 'Probability'},
            title="Prediction Confidence",
            color=list(prob_dict.keys()),
            color_discrete_map={
                'Very High': '#28a745',
                'High': '#007bff',
                'Medium': '#ffc107',
                'Low': '#dc3545'
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

# Information section
st.divider()
st.subheader("About HDI Tiers")

tier_info = pd.DataFrame({
    'Tier': ['Very High', 'High', 'Medium', 'Low'],
    'HDI Range': ['≥ 0.800', '0.700 - 0.799', '0.550 - 0.699', '< 0.550'],
    'Typical Characteristics': [
        'Excellent healthcare, high education, strong economy',
        'Good healthcare and education, developing economy',
        'Moderate healthcare and education, emerging economy',
        'Limited healthcare and education, developing economy'
    ]
})

st.table(tier_info)

st.subheader("How HDI is Calculated")
st.markdown("""
The Human Development Index is calculated using three dimensions:

1. **Health**: Measured by life expectancy at birth
2. **Education**: Measured by mean years of schooling and expected years of schooling
3. **Standard of Living**: Measured by Gross National Income (GNI) per capita (PPP)

This model uses machine learning to predict HDI tiers based on these key indicators.
""")
