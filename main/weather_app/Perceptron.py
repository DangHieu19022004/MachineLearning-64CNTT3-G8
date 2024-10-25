import base64
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder

# Đọc tệp dữ liệu
data = pd.read_csv("./seattle-weather.csv")

# Lọc dữ liệu
data = data.dropna()

# Các cột đặc trưng để huấn luyện mô hình
features = ["precipitation", "temp_max", "temp_min", "wind"]

# Chia dữ liệu thành X và y
X = data[features]
y = data["weather"]

# Mã hóa nhãn
le = LabelEncoder()
y = le.fit_transform(y)


# Chia dữ liệu thành 70% huấn luyện, 15% xác thực, 15% kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Khởi tạo và huấn luyện mô hình Perceptron
perceptron_model = Perceptron(max_iter=100, random_state=42, eta0=0.01, tol=0.001)
perceptron_model.fit(X_train, y_train)

def predict_weather(new_input):
    # new_input_pca = pca.transform(new_input)
    predicted_encoded = perceptron_model.predict(new_input)
    predicted_labels = le.inverse_transform(predicted_encoded)
    return predicted_labels

# Lưu mô hình và kết quả
valueSend = {
    'model': perceptron_model
}

# Save the model
joblib.dump(valueSend, 'perceptron_model.pkl')
