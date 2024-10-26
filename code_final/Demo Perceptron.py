# Nhập các thư viện cần thiết
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Tải bộ dữ liệu
data = pd.read_csv('./seattle-weather.csv')

# Tiền xử lý dữ liệu
data = data.dropna()  # Loại bỏ các giá trị thiếu
data = data.drop(['date'], axis=1)  # Loại bỏ cột không cần thiết

# Chia bộ dữ liệu thành đặc trưng (X) và nhãn (y)
X = data[["precipitation", "temp_max", "temp_min", "wind"]]
y = data["weather"]

# Chia dữ liệu thành 70% huấn luyện, 15% xác thực và 15% kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Chuẩn hóa đặc trưng
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Xác định k-fold StratifiedKFold để đảm bảo cân bằng lớp
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Khởi tạo mô hình Perceptron với các tham số đã tìm được
best_model = Perceptron(max_iter=1000, eta0=0.01, penalty='l2', alpha=0, tol=1e-4, random_state=42)

# Huấn luyện mô hình tốt nhất trên toàn bộ tập huấn luyện
best_model.fit(X_train, y_train)


# Lưu mô hình sử dụng joblib
valueSend = {
    'model': best_model,
}

joblib.dump(valueSend, 'best_model.pkl')
