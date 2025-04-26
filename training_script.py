import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
df = pd.read_csv("C:/Users/amrit/Placement/ML PROJECT/data/consistent_planarian_regeneration_dataset (1).csv")

# Preprocess data
X = df.drop(columns=['Regeneration_Rate'])
y = df['Regeneration_Rate']
X_encoded = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Save the scaler
joblib.dump(scaler, 'models/scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, 'models/rf_model.pkl')

# Save model features for encoding
joblib.dump(X_encoded.columns, 'models/model_features.pkl')