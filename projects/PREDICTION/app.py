from flask import Flask, request, jsonify
from joblib import load
from flask_cors import CORS  # Import CORS
import numpy as np  # Import numpy for np.array()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model (ensure this file is in the same directory as your app.py)
model = load('linear_regression_model.joblib')

@app.route('/')
def index():
    return "Welcome to the Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the frontend
    data = request.get_json()
    battery = data['battery']
    efficiency = data['efficiency']
    fast_charge = data['fast_charge']
    price_de = data['price_de']
    top_speed = data['top_speed']
    acceleration = data['acceleration']

    # Prepare the data for prediction (assuming it's in the same format as training data)
    input_data = np.array([[battery, efficiency, fast_charge, price_de, top_speed, acceleration]])

    # Predict the range using the trained model
    predicted_range = model.predict(input_data)

    # Return the prediction result as JSON
    return jsonify({'predicted_range': predicted_range[0]})

if __name__ == '__main__':
    app.run(debug=True)
