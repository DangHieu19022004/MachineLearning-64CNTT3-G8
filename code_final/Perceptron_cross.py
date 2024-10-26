# Nhập các thư viện cần thiết
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, hinge_loss)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler

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

# Khởi tạo mô hình Perceptron
perceptron_model = Perceptron(max_iter=1000, random_state=42)

# Định nghĩa grid các tham số để tìm kiếm bằng GridSearchCV
param_grid = {
    'max_iter': [1000, 2000, 3000],  # Tăng số vòng lặp để đảm bảo hội tụ
    'tol': [1e-4, 1e-5], 
    'eta0': [0.1, 0.01],  
    'penalty': ['l2', 'elasticnet'],  # Thêm điều chuẩn để tránh overfitting
    'alpha': [0.01, 0.1, 1,0]  # Thêm regularization
}

# Sử dụng GridSearchCV để tìm các tham số tốt nhất
grid_search = GridSearchCV(estimator=perceptron_model, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

# Huấn luyện với cross-validation
grid_search.fit(X_train, y_train)

# Xuất ra các tham số tốt nhất
print(f"Các tham số tốt nhất tìm thấy: {grid_search.best_params_}")
print(f"Độ chính xác cross-validation tốt nhất: {grid_search.best_score_:.2f}")

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_

# Huấn luyện mô hình tốt nhất trên toàn bộ tập huấn luyện
best_model.fit(X_train, y_train)

