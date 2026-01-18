%%writefile app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and scaler (this will be done once when the app starts)
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return 'Welcome to the Diabetes Prediction API!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        # Convert input JSON to a DataFrame for scaling
        query_df = pd.DataFrame(json_, index=[0])

        # Ensure columns are in the same order as during training
        # This assumes X.columns was derived from the training data columns
        # If X.columns is not available, we need to manually specify the order
        # For now, we'll assume the input JSON matches the original feature order
        # Define the expected columns based on the training data
        expected_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        query_df = query_df[expected_columns]

        # Scale the input features
        query_scaled = scaler.transform(query_df)

        # Make prediction
        prediction = model.predict(query_scaled)

        # Return prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}) , 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
