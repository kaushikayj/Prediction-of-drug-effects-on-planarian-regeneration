# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the dataset from CSV
df = pd.read_csv('C:/Users/amrit/Placement/ML PROJECT/data/consistent_planarian_regeneration_dataset (1).csv')  # Replace with your dataset path

# Visualize the distribution of the target variable
sns.countplot(x='Observed_Effect', data=df)
plt.title("Distribution of Observed Effect")
plt.show()

# Data Preprocessing
numeric_columns = df.select_dtypes(include=[np.number]).columns
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
df[non_numeric_columns] = df[non_numeric_columns].fillna('Unknown')

# Define Features (X) and Target (y)
X = df.drop(columns=['Observed_Effect'])
y = df['Observed_Effect']

# Encode the target variable (if it's categorical)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encoding for non-numeric columns (e.g., 'Drug')
X_encoded = pd.get_dummies(X, drop_first=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 1. Logistic Regression Model
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Metrics for Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLogistic Regression Metrics:")
print(f"Accuracy: {accuracy_lr * 100:.2f}%")
print(f"Mean Squared Error (MSE): {mse_lr:.2f}")
print(f"R² Score: {r2_lr:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))

# 2. Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# Metrics for Random Forest Classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Classifier Metrics:")
print(f"Accuracy: {accuracy_rf * 100:.2f}%")
print(f"Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"R² Score: {r2_rf:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))