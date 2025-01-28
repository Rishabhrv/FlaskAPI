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
from email.mime.text import MIMEText
import smtplib
import psutil 

# Initialize Flask app
app = Flask(__name__)


CONFIDENCE_THRESHOLD = 0.80  # 80% confidence threshold
CLOUD_API_URL = 'https://flaskapi-itlt.onrender.com/predict'

# Load the quantized Random Forest ONNX model
quantized_model_path = 'quantized_random_forest.onnx'
onnx_session = InferenceSession(quantized_model_path)
scaler = joblib.load('scaler.pkl')

# Load the encryption key (generate this once and keep it secure)
key = b'sU51iummF3ERq9MeIXa9C14ma1guxWFH12IyPTmZXTs='  
cipher = Fernet(key)

# Middleware to set start_time
@app.before_request
def start_timer():
    request.start_time = time.time()

# Email notification function
def send_email_notification(patient_id, doctor_email):
    subject = "Critical Alert: Heart Failure Detected"
    body = f"Patient ID {patient_id} is predicted to be at high risk of heart failure. Immediate action is required."
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'alerts@hospital.com'
    msg['To'] = doctor_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('your_email@gmail.com', 'your_password')
            server.sendmail('alerts@hospital.com', doctor_email, msg.as_string())
            print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Measure initial memory and CPU usage
        initial_memory = psutil.virtual_memory().used / (1024 ** 2)  # In MB
        initial_cpu = psutil.cpu_percent(interval=None)

        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if not file or not file.filename.endswith('.enc.gz'):
            return jsonify({"error": "Please upload a valid compressed and encrypted file (*.enc.gz)"}), 400

        # Measure encryption/decryption time
        decryption_start_time = time.time()
        try:
            encrypted_data = file.read()
            decompressed_data = gzip.decompress(encrypted_data)
            decrypted_data = cipher.decrypt(decompressed_data).decode('utf-8')
            data = pd.read_csv(StringIO(decrypted_data))
        except Exception as e:
            return jsonify({"error": f"Failed to decrypt or read the file: {str(e)}"}), 400
        decryption_end_time = time.time()
        decryption_time = decryption_end_time -decryption_start_time

        X = data.drop('target', axis=1)
        X_scaled = scaler.transform(X)

        input_data = X_scaled.astype(np.float32)

        if not isinstance(input_data, np.ndarray) or input_data.dtype != np.float32:
            return jsonify({"error": "Input data format error"}), 400

        input_name = onnx_session.get_inputs()[0].name
        input_dict = {input_name: input_data}

        # Measure inference time
        inference_start_time = time.time()
        predictions = onnx_session.run(None, input_dict)
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time

        confidence_scores = predictions[1][:, 1].tolist()

        low_confidence_count = sum(1 for score in confidence_scores if score < CONFIDENCE_THRESHOLD)

        if low_confidence_count > 0 and request.args.get('offload_to_cloud', 'false') == 'true':
            print("Offloading to cloud for further processing...")
            response = requests.post(CLOUD_API_URL, files=request.files, verify=False)
            cloud_response = response.json()
            cloud_response['cloud_offloaded'] = low_confidence_count
            return jsonify(cloud_response), 200

        critical_cases = []
        for idx, score in enumerate(confidence_scores):
            if score >= 0.95:
                critical_cases.append(idx)

        if critical_cases:
            send_email_notification(patient_id='12345', doctor_email='doctor@example.com')

        # Measure final memory and CPU usage
        final_memory = psutil.virtual_memory().used / (1024 ** 2)  # In MB
        final_cpu = psutil.cpu_percent(interval=None)

        response_data = {
            "confidence_scores": confidence_scores,
            "predictions": [1 if score >= CONFIDENCE_THRESHOLD else 0 for score in confidence_scores],
            "time_taken": time.time() - request.start_time,
            "critical_alerts": len(critical_cases),
            "cloud_offloaded": 0,
            "memory_usage_mb": final_memory - initial_memory,  # Memory used during processing
            "cpu_utilization_percent": final_cpu - initial_cpu,  # CPU used during processing
            "decryption_time_seconds": decryption_time,
            "inference_time_seconds": inference_time
        }
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(ssl_context=('cert.pem', 'key.pem'), debug=False)



