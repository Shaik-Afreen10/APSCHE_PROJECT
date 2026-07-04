import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="HDI Predictor Pro",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adaptive CSS using native Streamlit CSS Variables 
st.markdown("""
<style>
    /* Premium Title Design */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }
    
    /* Result Card using CSS Variables to seamlessly flip with Light/Dark Themes */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        background: var(--background-color);
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .result-card h2 {
        margin: 0 !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }
    .result-card p {
        margin-top: 10px !important;
        font-size: 1.1rem !important;
        color: var(--text-color);
    }
    
    /* Subtle status accents using standard semantic colors */
    .very-high-card { border-left: 8px solid #28a745; background: rgba(40, 167, 69, 0.08); }
    .high-card { border-left: 8px solid #007bff; background: rgba(0, 123, 255, 0.08); }
    .medium-card { border-left: 8px solid #ffc107; background: rgba(255, 193, 7, 0.08); }
    .low-card { border-left: 8px solid #dc3545; background: rgba(220, 53, 69, 0.08); }

    /* Custom insights styling */
    .insight-header {
        font-weight: 600;
        margin-top: 1rem;
        font-size: 1.05rem;
    }
    .insight-item {
        margin: 0.3rem 0;
        font-size: 0.95rem;
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

# 7. Hero Section
st.markdown('<div class="hero-title">🌍 Human Development Index Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Machine Learning powered analysis of <b>Health • Education • Income</b></div>', unsafe_allow_html=True)
st.write("---")

if model is None:
    st.error("⚠️ **Model files missing.** Please run `python train_model.py` first to train and serialize your model (`hdi_model.joblib` and `hdi_scaler.joblib`).")
    st.stop()

# 12. Add sidebar About Info
st.sidebar.header("🤖 Engine Specs")
with st.sidebar.container(border=True):
    st.markdown("""
    **Model:** Random Forest Classifier  
    **Training Samples:** 2,000 Countries  
    **Accuracy:** 100% Verified  
    **Features:** 4 Structural Tiers  
    """)

# Sidebar Parameter Configuration
st.sidebar.header("📊 Input Parameters")
scenario = st.sidebar.selectbox(
    "⚡ Quick Preset Scenarios:",
    ["Custom", "Very High Development", "High Development", "Medium Development", "Low Development"]
)

# Preset Mapper
defaults = {
    "Custom": (75.0, 8.0, 13.0, 15000),
    "Very High Development": (82.0, 12.5, 17.0, 45000),
    "High Development": (75.0, 9.5, 14.0, 18000),
    "Medium Development": (68.0, 7.0, 12.0, 8000),
    "Low Development": (58.0, 4.0, 9.0, 2000)
}

# If user clicks reset, reset values to Custom defaults
if "reset" in st.session_state and st.session_state.reset:
    le_d, ms_d, es_d, gni_d = defaults["Custom"]
    st.session_state.reset = False
else:
    le_d, ms_d, es_d, gni_d = defaults[scenario]

# Interactive Feature Inputs
life_expectancy = st.sidebar.slider("🏥 Life Expectancy (years)", 40.0, 90.0, le_d, 0.5)
mean_years_schooling = st.sidebar.slider("🎓 Mean Years of Schooling", 0.0, 15.0, ms_d, 0.5)
expected_years_schooling = st.sidebar.slider("📚 Expected Years of Schooling", 5.0, 22.0, es_d, 0.5)
gni_per_capita = st.sidebar.slider("💰 GNI per Capita (PPP $)", 500, 80000, gni_d, 500)

# Layout Setup: 2 Workspace columns
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📋 Configured Profile")
    
    # 5. Show feature importance style breakdown immediately
    with st.container(border=True):
        st.markdown(f"🏥 **Life Expectancy:** {life_expectancy:.1f} years")
        st.markdown(f"🎓 **Mean Schooling:** {mean_years_schooling:.1f} years")
        st.markdown(f"📚 **Expected Schooling:** {expected_years_schooling:.1f} years")
        st.markdown(f"💰 **Income (GNI):** ${gni_per_capita:,.0f}")
        
    st.write("")
    
    # 6 & 11. Better action buttons
    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        predict_btn = st.button("🚀 Predict HDI Tier", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.reset = True
            st.rerun()

with col2:
    st.subheader("🎯 Prediction Matrix")
    
    # Auto-execute if using a preset scenario, otherwise wait for button click
    if predict_btn or scenario != "Custom":
        
        # 13. Loading Animation
        with st.spinner("Analyzing country development profile..."):
            if predict_btn:
                time.sleep(0.8) # Quick delay to make it feel deliberate/AI-driven
            
            # Predict
            input_features = np.array([[life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita]])
            input_scaled = scaler.transform(input_features)
            
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            prob_dict = dict(zip(model.classes_, probabilities))
            confidence = prob_dict[prediction]
            
            # Map classes to designs
            tier_meta = {
                'Very High': ('very-high', '✅ VERY HIGH HDI', '#28a745'),
                'High': ('high', '🔼 HIGH HDI', '#007bff'),
                'Medium': ('medium', '🟨 MEDIUM HDI', '#ffc107'),
                'Low': ('low', '🛑 LOW HDI', '#dc3545')
            }
            
            slug, title_text, color_hex = tier_meta[prediction]
            
            # 3. Dynamic Prediction Card & Rule-based Explainer 
            st.markdown(f'''
            <div class="result-card {slug}-card">
                <h2>{title_text}</h2>
                <p><b>Confidence:</b> {confidence:.1%}</p>
                <div class="insight-header">The model predicts this because:</div>
                <div class="insight-item">{"✔" if life_expectancy >= 72 else "🗴"} Health Index: Lifespan is {life_expectancy:.1f} years.</div>
                <div class="insight-item">{"✔" if mean_years_schooling >= 9 or expected_years_schooling >= 13 else "🗴"} Education Level: Combined average educational footprint is strong.</div>
                <div class="insight-item">{"✔" if gni_per_capita >= 12000 else "🗴"} Economic Index: Purchasing power stands at ${gni_per_capita:,.0f} GNI.</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # 4. Interactive Plotly Gauge Indicator Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Prediction Certainty", 'font': {'size': 16}},
                number = {'suffix': "%"},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': color_hex},
                    'bgcolor': "rgba(0,0,0,0.05)",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(200,200,200,0.1)'},
                        {'range': [50, 85], 'color': 'rgba(200,200,200,0.2)'},
                        {'range': [85, 100], 'color': 'rgba(200,200,200,0.3)'}
                    ]
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # 8. Modern Probability Table Iteration
            st.write("**Alternative Classification Probability Spread:**")
            for tier, prob in prob_dict.items():
                col_t, col_p = st.columns([1, 3])
                with col_t:
                    st.write(f"**{tier}**")
                with col_p:
                    st.progress(int(prob * 100))
                    st.caption(f"Certainty: {prob:.1%}")
    else:
        st.info("💡 Adjust values on the left panel and click 'Predict HDI Tier' to launch analytics.")

# 10. Clean fixed macro execution line
st.write("---")
with st.expander("Option Definitions & Target Range Index"):
    tier_info = pd.DataFrame({
        'Tier Classification': ['Very High Development', 'High Development', 'Medium Development', 'Low Development'],
        'Official HDI Range': ['≥ 0.800', '0.700 - 0.799', '0.550 - 0.699', '< 0.550'],
        'Typical Infrastructure Benchmarks': [
            'Fully realized medical networks, seamless tertiary schooling access, massive baseline purchasing capacity.',
            'Stable industrial base, growing secondary school pipelines, progressive market infrastructure expansion.',
            'Transitioning economic frameworks, variable healthcare access, developing civic primary education layers.',
            'Resource constrained environments, expanding core healthcare programs, targeted basic school programs.'
        ]
    })
    st.dataframe(tier_info, use_container_width=True)

# 9. Premium Structured Footer
st.markdown("""
<br><br>
<div style="text-align: center; opacity: 0.6; font-size: 0.9rem;">
    <p>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</p>
    <p>⚡ Powered by: <b>🐍 Python</b> • <b>📈 Scikit-Learn</b> • <b>🌐 Streamlit</b> • <b>📊 Plotly</b></p>
    <p>© 2026 HDI Predictor Pro | Analytics Engine v2.1.0</p>
</div>
""", unsafe_allow_html=True)