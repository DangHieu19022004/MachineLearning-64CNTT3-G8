import numpy as np
import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the model
clf = joblib.load('weather_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.get_json()
    precipitation = data['precipitation']
    temp_max = data['temp_max']
    temp_min = data['temp_min']
    wind = data['wind']

    # Predict
    prediction = clf.predict(np.array([[precipitation, temp_max, temp_min, wind]]))

    # Return prediction result
    return jsonify({
        'prediction': prediction[0],
        'confidence': clf.predict_proba([[precipitation, temp_max, temp_min, wind]]).max()
    })

if __name__ == '__main__':
    app.run(debug=True)
