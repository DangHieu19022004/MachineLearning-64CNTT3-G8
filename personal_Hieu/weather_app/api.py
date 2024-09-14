import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.stats import entropy
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

app = Flask(__name__)

# Load the model
models = {
    # 'perceptron': joblib.load('perceptron_model.pkl'),
    'decision_tree': joblib.load('weather_model.pkl')
    # 'neural_network': joblib.load('neural_network_model.pkl')
}

# Control routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = request.get_json()
        algorithm = data['algorithm']
        model_data = models[algorithm]

        clf = model_data['model']
        accuracy = model_data['accuracy']
        report_after = model_data['report']
        plot_url = model_data['plot_url']
        entropy_url = model_data['entropy_url']

        precipitation = data['precipitation']
        temp_max = data['temp_max']
        temp_min = data['temp_min']
        wind = data['wind']

        # Predict
        prediction = clf.predict(np.array([[precipitation, temp_max, temp_min, wind]]))

        # Return prediction result
        return jsonify({
            'prediction': prediction[0],
            'confidence': accuracy,
            'report': f"<pre>{report_after}</pre>",
            'plot_url': plot_url,
            'entropy_url': entropy_url
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }),500


if __name__ == '__main__':
    app.run(debug=True)
