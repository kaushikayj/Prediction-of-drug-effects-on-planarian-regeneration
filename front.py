from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    input_data = pd.DataFrame([form_data])

    # Preprocess and scale input data
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    model_features = joblib.load('models/model_features.pkl')
    input_data_encoded = input_data_encoded.reindex(columns=model_features, fill_value=0)
    input_scaled = scaler.transform(input_data_encoded)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    return render_template('results.html', prediction=f'Predicted Regeneration Rate: {prediction:.2f}')

if __name__ == '__main__':
    app.run(debug=True)