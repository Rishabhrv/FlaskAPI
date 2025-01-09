from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle as pkl
import time
from io import StringIO
from cryptography.fernet import Fernet  
import xgboost

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
with open('xgboost.pkl', 'rb') as file:
    model = pkl.load(file)

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
        if not file or not file.filename.endswith('.enc'):
            return jsonify({"error": "Please upload a valid encrypted file (*.enc)"}), 400

        # Read and decrypt the file
        try:
            encrypted_data = file.read()
            decrypted_data = cipher.decrypt(encrypted_data).decode('utf-8')
            data = pd.read_csv(StringIO(decrypted_data))
        except Exception as e:
            return jsonify({"error": f"Failed to decrypt or read the file: {str(e)}"}), 400

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
            predictions.append(int(prediction))

        # Calculate processing time
        time_taken = time.time() - request.start_time

        return jsonify({
            "predictions": predictions,
            "time_taken": time_taken
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(ssl_context=('cert.pem', 'key.pem'), debug=False)
