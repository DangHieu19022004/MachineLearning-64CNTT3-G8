import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.stats import entropy
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

from NeuralNetWork import encode_input
from Perceptron import predict_weather

app = Flask(__name__)

# Load the model
models = {
    'perceptron': joblib.load('perceptron_model.pkl'),
    'decision_tree': joblib.load('decision_tree.pkl'),
    'neural_network': joblib.load('neural_network_model.pkl'),
    'stacking': joblib.load('stacking_model.pkl')
}

# Control routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        # Get data from the form
        data = request.get_json()
        algorithm = data['algorithm']
        model_data = models[algorithm]

        clf = model_data['model']

        #get data from input
        precipitation = float(data['precipitation'])
        temp_max = float(data['temp_max'])
        temp_min = float(data['temp_min'])
        wind = float(data['wind'])


        try:
            if algorithm == 'perceptron':
                prediction = predict_weather(np.array([[precipitation, temp_max, temp_min, wind]]))
                print(f"Prediction: {prediction}")
            elif algorithm == 'neural_network':
                prediction = encode_input(np.array([[precipitation, temp_max, temp_min, wind]]))
                print(f"Prediction: {prediction}")
            else:
                prediction = clf.predict(np.array([[precipitation, temp_max, temp_min, wind]]))

            return jsonify({
                'prediction': prediction.tolist()
            })
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Prediction failed'}), 500



if __name__ == '__main__':
    app.run(debug=True)
