from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle as pkl # or pickle, depending on how your model is saved
import time

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
with open('xgboost.pkl', 'rb') as file:
    model = pkl.load(file)  # Replace with your model's path

# Middleware to set start_time
@app.before_request
def start_timer():
    request.start_time = time.time()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is included in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        # Check if the file is a CSV
        if not file or not file.filename.endswith('.csv'):
            return jsonify({"error": "Please upload a valid CSV file"}), 400

        # Read the CSV file into a DataFrame
        try:
            data = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Failed to read CSV file: {str(e)}"}), 400

        # Ensure the 'target' column is present
        if 'target' not in data.columns:
            return jsonify({"error": "The uploaded file must include a 'target' column"}), 400

        # Prepare data for prediction
        X = data.drop('target', axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled)

        # Simulate continuous prediction
        predictions = []
        for index, row in X_scaled.iterrows():
            row_data = row.values.reshape(1, -1)
            prediction = model.predict(row_data)[0]
            predictions.append({"prediction": int(prediction)})  # Convert to int
            time.sleep(0.03)  # Simulating real-time prediction delay

        # Calculate processing time
        time_taken = time.time() - request.start_time

        return jsonify({
            "predictions": predictions,
            "time_taken": time_taken
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500 

if __name__ == '__main__':
    app.run(debug=False)