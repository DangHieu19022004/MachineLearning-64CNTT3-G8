# Nhập các thư viện cần thiết
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, hinge_loss)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
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

# Dự đoán trên các tập huấn luyện, xác thực và kiểm tra
y_train_pred = best_model.predict(X_train)
y_valid_pred = best_model.predict(X_valid)
y_test_pred = best_model.predict(X_test)

# Đánh giá mô hình trên tập kiểm tra
accuracy = accuracy_score(y_test, y_test_pred) * 100
print(f"Độ chính xác trên tập kiểm tra: {accuracy:.2f}%")

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.ylabel('Nhãn thật')
plt.xlabel('Nhãn dự đoán')
plt.title("Ma trận nhầm lẫn trên tập kiểm tra")

# Báo cáo phân loại
print("Báo cáo tập huấn luyện:")
print(classification_report(y_train, y_train_pred, zero_division=0))
print("Báo cáo tập xác thực:")
print(classification_report(y_valid, y_valid_pred, zero_division=0))
print("Báo cáo tập kiểm tra:")
print(classification_report(y_test, y_test_pred, zero_division=0))

# Đường học (Learning Curve) trên tất cả các tập dữ liệu (huấn luyện, xác thực, kiểm tra)
train_sizes, train_scores, valid_scores = learning_curve(
    best_model, X_train, y_train, cv=kf, scoring='accuracy',
    n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
)

# Tính trung bình và độ lệch chuẩn cho từng đường cong
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

# Tính độ chính xác cho tập kiểm tra ở từng kích thước của train_sizes
test_acc = []
for train_size in train_sizes:
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
    best_model.fit(X_train_subset, y_train_subset)
    test_acc.append(accuracy_score(y_test, best_model.predict(X_test)))

# Vẽ Learning Curve cho các tập huấn luyện, xác thực và kiểm tra
plt.figure()
plt.title('Learning Curve (Perceptron với tham số đã điều chỉnh và StratifiedKFold)')
plt.xlabel('Số lượng mẫu huấn luyện')
plt.ylabel('Điểm số')

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.1, color="g")

# Vẽ các đường học tập
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Điểm số tập huấn luyện")
plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Điểm số tập xác thực")
plt.plot(train_sizes, test_acc, 'o-', color="b", label="Điểm số tập kiểm tra")

plt.legend(loc="best")
plt.show()

# Bước 2: Vẽ biểu đồ Hinge Loss theo epochs
train_losses = []
best_model = Perceptron(max_iter=1, tol=None, random_state=42, warm_start=True)

# Huấn luyện lại mô hình với 100 epochs và tính toán hinge loss
for epoch in range(100):
    best_model.fit(X_train, y_train)
    y_train_pred_decision = best_model.decision_function(X_train)  # decision_function trả về các điểm tin cậy
    loss_value = hinge_loss(y_train, y_train_pred_decision)
    train_losses.append(loss_value)

# Vẽ biểu đồ Hinge Loss theo epochs
plt.figure(figsize=(10, 6))
plt.plot(range(100), train_losses, label='Hinge Loss')
plt.xlabel('Số lượng epochs')
plt.ylabel('Hinge Loss')
plt.title('Hinge Loss qua các Epochs (Perceptron)')
plt.legend()
plt.grid(True)
plt.show()


valueSend = {
    'model': best_model,
}

joblib.dump(valueSend, 'best_model.pkl')
