import base64
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Nhận dữ liệu
data = pd.read_csv('./seattle-weather.csv')
data = data.dropna()
data.drop(['date'], axis=1, inplace=True)

# Mã hóa nhãn
le = LabelEncoder()
data['weather_encoded'] = le.fit_transform(data['weather'])

# Chia dữ liệu
X = data[["precipitation", "temp_max", "temp_min", "wind"]]
y = data['weather_encoded']

# Chia dữ liệu thành 70% huấn luyện, 15% xác thực, 15% kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Khởi tạo các mô hình cơ bản với điều chỉnh
base_learners = [
    ('decision_tree', tree.DecisionTreeClassifier(criterion='entropy', random_state=42,
                                    class_weight = None,
                                    max_depth = 3,
                                    max_features = 'sqrt',
                                    max_leaf_nodes = None,
                                    min_samples_leaf = 1
                                    , min_samples_split = 2,
                                    min_weight_fraction_leaf = 0.0,
                                    splitter = 'best')),  # Tăng độ sâu một chút
    ('perceptron', Perceptron(max_iter=100, random_state=42, eta0=0.01, tol=0.001)),
    ('neural_network', MLPClassifier(hidden_layer_sizes=(40, 20, 10),
                                    max_iter=500,
                                    activation='tanh',
                                    solver='adam',
                                    random_state=42,
                                    early_stopping=True,
                                    learning_rate_init=0.01))  # Giảm số lượng nút trong lớp ẩn và thêm regularization
]

# Khởi tạo mô hình Stacking
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=MLPClassifier(hidden_layer_sizes=(5,), alpha=0.01, max_iter=1000, random_state=42))

# Huấn luyện mô hình Stacking
stacking_model.fit(X_train, y_train)

valueSend = {
    'model': stacking_model,
}

joblib.dump(valueSend, 'stacking_model.pkl')
