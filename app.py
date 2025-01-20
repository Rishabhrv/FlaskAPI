import time
import gzip
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from cryptography.fernet import Fernet
from onnxruntime import InferenceSession
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from io import StringIO
import requests

# Initialize Flask app
app = Flask(__name__)


CONFIDENCE_THRESHOLD = 0.80  # 80% confidence threshold
CLOUD_API_URL = 'https://flaskapi-itlt.onrender.com/predict'

# Load the quantized Random Forest ONNX model
quantized_model_path = 'quantized_random_forest.onnx'
onnx_session = InferenceSession(quantized_model_path)

# Load the encryption key (generate this once and keep it secure)
key = b'sU51iummF3ERq9MeIXa9C14ma1guxWFH12IyPTmZXTs='  
cipher = Fernet(key)

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

        # Check if the file is uploaded and is encrypted
        if not file or not file.filename.endswith('.enc.gz'):
            return jsonify({"error": "Please upload a valid compressed and encrypted file (*.enc.gz)"}), 400

        # Read, decompress and decrypt the file
        try:
            encrypted_data = file.read()
            decompressed_data = gzip.decompress(encrypted_data)
            decrypted_data = cipher.decrypt(decompressed_data).decode('utf-8')
            data = pd.read_csv(StringIO(decrypted_data))
        except Exception as e:
            return jsonify({"error": f"Failed to decrypt or read the file: {str(e)}"}), 400

        # Prepare data for prediction
        X = data.drop('target', axis=1)

        # Load the saved scaler object
        scaler = joblib.load('scaler.pkl')
        X_scaled = scaler.transform(X)  # Use transform instead of fit_transform

        # Convert data to the correct format for ONNX model (float32)
        input_data = X_scaled.astype(np.float32)

        if not isinstance(input_data, np.ndarray) or input_data.dtype != np.float32:
            return jsonify({"error": "Input data format error"}), 400
        
        # Run inference with the ONNX model
        input_name = onnx_session.get_inputs()[0].name
        input_dict = {input_name: input_data}
        predictions = onnx_session.run(None, input_dict)

        # Extract confidence scores (assuming binary classification)
        confidence_scores = predictions[1][:, 1].tolist()

        low_confidence_count = sum(1 for score in confidence_scores if score < CONFIDENCE_THRESHOLD)

        # Decision Criteria for Edge-Cloud Collaboration
        if low_confidence_count > 0 and request.args.get('offload_to_cloud', 'false') == 'true':
            print("Offloading to cloud for further processing...")

            response = requests.post(CLOUD_API_URL, files=request.files, verify=False)
            cloud_response = response.json()
            cloud_response['cloud_offloaded'] = low_confidence_count  # Track how many were offloaded
            return jsonify(cloud_response), 200


        # # Check if predictions contain a list of probabilities or a single scalar
        # if isinstance(predictions[0], (list, np.ndarray)):
        #     confidence_scores = [float(p) for p in predictions[0]]  # Convert to float for JSON serialization
        # else:
        #     confidence_scores = [float(predictions[0])]  # Convert single scalar value to list

        response_data = {
            "confidence_scores": confidence_scores,
            "predictions": [1 if score >= CONFIDENCE_THRESHOLD else 0 for score in confidence_scores],
            "time_taken": time.time() - request.start_time,
            "cloud_offloaded": 0
        }
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(ssl_context=('cert.pem', 'key.pem'), debug=False)
