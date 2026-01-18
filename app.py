from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained model artifacts
loaded_model = joblib.load("logistic_regression_model.joblib")
loaded_scaler = joblib.load("standard_scaler.joblib")
feature_names = joblib.load("feature_names.joblib")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([data], columns=feature_names)

        # Scale input
        scaled_input = loaded_scaler.transform(input_df)

        # Predict
        prediction = loaded_model.predict(scaled_input)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
