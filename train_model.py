import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic HDI training data based on real HDI calculation methodology."""
    np.random.seed(42)
    
    # Generate features based on HDI tiers
    data = []
    
    # Very High HDI countries (approx 20% of data)
    for _ in range(int(n_samples * 0.2)):
        life_expectancy = np.random.uniform(75, 89)
        mean_schooling = np.random.uniform(10, 14)
        expected_schooling = np.random.uniform(15, 20)
        gni_per_capita = np.random.uniform(25000, 80000)
        hdi_tier = 'Very High'
        data.append([life_expectancy, mean_schooling, expected_schooling, gni_per_capita, hdi_tier])
    
    # High HDI countries (approx 30% of data)
    for _ in range(int(n_samples * 0.3)):
        life_expectancy = np.random.uniform(70, 78)
        mean_schooling = np.random.uniform(8, 11)
        expected_schooling = np.random.uniform(12, 16)
        gni_per_capita = np.random.uniform(12000, 25000)
        hdi_tier = 'High'
        data.append([life_expectancy, mean_schooling, expected_schooling, gni_per_capita, hdi_tier])
    
    # Medium HDI countries (approx 30% of data)
    for _ in range(int(n_samples * 0.3)):
        life_expectancy = np.random.uniform(60, 72)
        mean_schooling = np.random.uniform(5, 9)
        expected_schooling = np.random.uniform(10, 14)
        gni_per_capita = np.random.uniform(4000, 12000)
        hdi_tier = 'Medium'
        data.append([life_expectancy, mean_schooling, expected_schooling, gni_per_capita, hdi_tier])
    
    # Low HDI countries (approx 20% of data)
    for _ in range(int(n_samples * 0.2)):
        life_expectancy = np.random.uniform(50, 65)
        mean_schooling = np.random.uniform(2, 6)
        expected_schooling = np.random.uniform(7, 12)
        gni_per_capita = np.random.uniform(500, 4000)
        hdi_tier = 'Low'
        data.append([life_expectancy, mean_schooling, expected_schooling, gni_per_capita, hdi_tier])
    
    df = pd.DataFrame(data, columns=[
        'life_expectancy', 
        'mean_years_schooling', 
        'expected_years_schooling', 
        'gni_per_capita',
        'hdi_tier'
    ])
    
    return df

def train_model():
    """Train and save the HDI prediction model."""
    print("Generating synthetic training data...")
    df = generate_synthetic_data(2000)
    
    # Save training data for reference
    df.to_csv('hdi_training_data.csv', index=False)
    print(f"Training data saved: {len(df)} samples")
    
    # Prepare features and target
    X = df[['life_expectancy', 'mean_years_schooling', 'expected_years_schooling', 'gni_per_capita']]
    y = df['hdi_tier']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Classifier
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    joblib.dump(model, 'hdi_model.joblib')
    joblib.dump(scaler, 'hdi_scaler.joblib')
    print("\nModel and scaler saved successfully!")
    
    return model, scaler, accuracy

if __name__ == "__main__":
    train_model()
