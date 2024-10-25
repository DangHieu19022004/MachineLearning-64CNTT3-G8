import base64
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

# Đọc dữ liệu từ file CSV
df = pd.read_csv("./seattle-weather.csv")

# Xử lý dữ liệu thiếu (nếu có) bằng cách loại bỏ các hàng có giá trị null
df = df.dropna()

# Mã hóa thuộc tính weather (categorical) thành số
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])

# Sử dụng các cột numerical như precipitation, temp_max, temp_min, wind làm dữ liệu đầu vào
X_data = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values

# Chuẩn hóa (scale) các thuộc tính numerical để phù hợp với MLP
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

# Nhãn (target) là cột weather đã được mã hóa
y_data = df['weather_encoded'].values

# Chia tập dữ liệu thành 3 tập: training (70%), validation (15%), test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.3, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

# Lấy một phần dữ liệu huấn luyện (giảm kích thước)
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Xây dựng mô hình MLP (Multilayer Perceptron - Neural Network) với Early Stopping
clf = MLPClassifier(hidden_layer_sizes=(40, 20, 10),
    max_iter=500,
    activation='tanh',
    solver='adam',
    random_state=42,
    early_stopping=True,
    learning_rate_init=0.01)

# Huấn luyện mô hình
clf.fit(X_train_sample, y_train_sample)

def encode_input(new_input):
    predicted_encoded = clf.predict(new_input)
    predicted_labels = le.inverse_transform(predicted_encoded)
    return predicted_labels

valueSend = {
    'model': clf
}

# Save the model
joblib.dump(valueSend, 'neural_network_model.pkl')
