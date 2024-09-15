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
    'decision_tree': joblib.load('decision_tree.pkl'),
    'neural_network': joblib.load('neural_network_model.pkl')
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
        accuracy = model_data['accuracy']
        report_after = model_data['report']
        plot_url = model_data['plot_url']
        entropy_url = model_data['entropy_url']

        #get data from input
        precipitation = float(data['precipitation'])
        temp_max = float(data['temp_max'])
        temp_min = float(data['temp_min'])
        wind = float(data['wind'])


        try:
            prediction = clf.predict(np.array([[precipitation, temp_max, temp_min, wind]]))
            return jsonify({
                'prediction': prediction.tolist(),
                'confidence': accuracy,
                'report': f"<pre>{report_after}</pre>",
                'plot_url': plot_url,
                'entropy_url': entropy_url
            })
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Prediction failed'}), 500



if __name__ == '__main__':
    app.run(debug=True)
