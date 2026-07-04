# HDI Predictor - Human Development Index Classification

A machine learning-powered web application that predicts a country's Human Development Index (HDI) tier based on key development indicators: life expectancy, education, and income.

## Features

- **Interactive Web Interface**: Built with Streamlit for easy use
- **ML-Based Predictions**: Uses Random Forest classifier trained on synthetic data
- **Quick Scenarios**: Pre-configured scenarios for Very High, High, Medium, and Low development
- **Visual Results**: Probability charts and detailed explanations
- **Real-time Predictions**: Instant feedback as you adjust parameters

## HDI Tiers

The model classifies countries into four HDI tiers:

- **Very High** (≥ 0.800): Excellent healthcare, high education, strong economy
- **High** (0.700 - 0.799): Good healthcare and education, developing economy
- **Medium** (0.550 - 0.699): Moderate healthcare and education, emerging economy
- **Low** (< 0.550): Limited healthcare and education, developing economy

## Installation

1. Navigate to the project directory:
```bash
cd hdi_predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### Step 1: Train the Model

First, train the machine learning model with synthetic data:

```bash
python train_model.py
```

This will:
- Generate 2000 synthetic training samples
- Train a Random Forest classifier
- Save the model as `hdi_model.joblib`
- Save the scaler as `hdi_scaler.joblib`
- Save training data as `hdi_training_data.csv`
- Display model accuracy and classification report

Expected output:
```
Generating synthetic training data...
Training data saved: 2000 samples
Training Random Forest model...
Model accuracy: 1.000
...
Model and scaler saved successfully!
```

### Step 2: Launch the Web Application

Run the Streamlit application:

```bash
python -m streamlit run app.py --server.headless true
```

The application will start at `http://localhost:8501`

**Note:** If you don't want to use headless mode, simply run:
```bash
python -m streamlit run app.py
```

## Using the Application

### Input Parameters (Left Sidebar)

Adjust these sliders to set country development indicators:

1. **🏥 Life Expectancy (years)**
   - Range: 40 to 90 years
   - Default: 75 years
   - What it means: Average number of years a person is expected to live
   - Higher values indicate better healthcare systems

2. **🎓 Mean Years of Schooling**
   - Range: 0 to 15 years
   - Default: 8 years
   - What it means: Average years of education for adults aged 25+
   - Higher values indicate better education systems

3. **📚 Expected Years of Schooling**
   - Range: 5 to 22 years
   - Default: 13 years
   - What it means: Years of schooling a child can expect to receive
   - Higher values indicate better educational opportunities

4. **💰 GNI per Capita (PPP $)**
   - Range: $500 to $80,000
   - Default: $15,000
   - What it means: Gross National Income per person adjusted for cost of living
   - Higher values indicate stronger economies

### Quick Scenarios

Select from pre-configured scenarios to auto-fill values:

- **Very High Development**: Life: 82 yrs, Schooling: 12.5/17 yrs, GNI: $45,000
- **High Development**: Life: 75 yrs, Schooling: 9.5/14 yrs, GNI: $18,000
- **Medium Development**: Life: 68 yrs, Schooling: 7/12 yrs, GNI: $8,000
- **Low Development**: Life: 58 yrs, Schooling: 4/9 yrs, GNI: $2,000

### Making Predictions

1. Adjust the sliders OR select a quick scenario
2. Click the **"Predict HDI Tier"** button
3. View results on the right side:
   - Predicted HDI tier with color-coded box
   - Detailed explanation of the prediction
   - Probability breakdown for all tiers
   - Visual confidence chart

## Example Input Values

### For a Developed Country (Very High HDI)
- Life Expectancy: 80-85 years
- Mean Years of Schooling: 11-14 years
- Expected Years of Schooling: 16-19 years
- GNI per Capita: $30,000-$60,000

### For a Developing Country (Medium HDI)
- Life Expectancy: 65-72 years
- Mean Years of Schooling: 6-9 years
- Expected Years of Schooling: 11-14 years
- GNI per Capita: $5,000-$12,000

### For a Least Developed Country (Low HDI)
- Life Expectancy: 50-65 years
- Mean Years of Schooling: 2-6 years
- Expected Years of Schooling: 7-12 years
- GNI per Capita: $500-$4,000

## Project Structure

```
hdi_predictor/
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── hdi_model.joblib           # Trained model (generated after training)
├── hdi_scaler.joblib          # Feature scaler (generated after training)
└── hdi_training_data.csv      # Synthetic training data (generated after training)
```

## Technical Details

- **Model**: Random Forest Classifier (100 estimators, max depth 10)
- **Features**: Life expectancy, mean years of schooling, expected years of schooling, GNI per capita
- **Training Data**: 2000 synthetic samples based on real HDI methodology
- **Accuracy**: 100% on test set
- **Framework**: Streamlit for UI, scikit-learn for ML

## Use Cases

### Scenario 1: Predicting Very High Human Development
A user selects a country with high life expectancy (82 years), strong mean years of schooling (12.5), expected years of schooling (17), and high GNI per capita ($45,000). The model predicts a Very High HDI score.

### Scenario 2: Identifying Development Gaps in Emerging Economies
A policymaker inputs mid-range values: life expectancy (68 years), average educational attainment (7 years), and moderate income ($8,000). The model returns a Medium HDI score, highlighting areas for improvement.

### Scenario 3: Assessing Countries Requiring Development Intervention
A researcher evaluates a country with low life expectancy (58 years), limited educational opportunities (4 years), and low GNI per capita ($2,000). The model predicts a Low HDI score, indicating developmental challenges.

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- plotly

## Notes

- The training data is synthetic and generated based on real HDI calculation methodology
- For production use, replace with actual HDI data from UNDP reports
- The model is for educational and demonstration purposes
