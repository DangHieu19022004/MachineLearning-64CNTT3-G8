import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

# Bước 1: Nhận dữ liệu
df = pd.read_csv('./seattle-weather.csv').dropna()

# Bước 2: Mã hóa nhãn
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])

# Bước 3: Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X_data = scaler.fit_transform(df[['precipitation', 'temp_max', 'temp_min', 'wind']].values)
y_data = df['weather_encoded'].values

# Bước 4: Chia dữ liệu thành 70% huấn luyện, 15% xác thực, 15% kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.3, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

# Bước 5: Huấn luyện mô hình Neural Network trước khi xử lý
clf_before = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=500, activation='relu', solver='adam',
                            random_state=42, early_stopping=True, learning_rate_init=0.01)
clf_before.fit(X_train, y_train)

# Bước 6: Dự đoán và đánh giá mô hình trước khi xử lý
y_train_pred_before = clf_before.predict(X_train)
y_val_pred_before = clf_before.predict(X_val)
y_test_pred_before = clf_before.predict(X_test)

def evaluate_model(y_true, y_pred, dataset_name):
    print(f"Đánh giá mô hình trên {dataset_name}:")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")

evaluate_model(y_train, y_train_pred_before, "tập huấn luyện")
evaluate_model(y_val, y_val_pred_before, "tập xác thực")
evaluate_model(y_test, y_test_pred_before, "tập kiểm tra")

# Bước 7: Cross-validation
cv_scores_before = cross_val_score(clf_before, X_train, y_train, cv=5)
print(f"Cross-validation scores (before processing): {cv_scores_before}")
print(f"Mean cross-validation score (before processing): {np.mean(cv_scores_before):.4f}\n")

# Bước 8: Đồ thị phân tán chưa chuẩn hóa
plt.figure(figsize=(12, 6))  # Adjust the figure size

# Đồ thị phân tán dữ liệu gốc cho Nhiệt độ Tối Đa và Nhiệt độ Tối Thiểu
plt.subplot(1, 2, 1)  # Chỉ cần 2 cột
plt.scatter(df['temp_max'], df['temp_min'], c=df['weather_encoded'], cmap='viridis', alpha=0.7, s=50)  # Increased marker size
plt.title('Nhiệt độ Max vs Nhiệt độ Min', fontsize=16)  # Enhanced title font size
plt.xlabel('Nhiệt độ Tối Đa', fontsize=14)  # Enhanced label font size
plt.ylabel('Nhiệt độ Tối Thiểu', fontsize=14)  # Enhanced label font size
plt.colorbar(label='Lớp Thời Tiết')
plt.grid(True)  # Add grid for better readability

# Đồ thị phân tán dữ liệu gốc cho Lượng Mưa và Gió
plt.subplot(1, 2, 2)  # Chỉ cần 2 cột
plt.scatter(df['precipitation'], df['wind'], c=df['weather_encoded'], cmap='viridis', alpha=0.7, s=50)  # Increased marker size
plt.title('Lượng Mưa vs Gió', fontsize=16)  # Enhanced title font size
plt.xlabel('Lượng Mưa', fontsize=14)  # Enhanced label font size
plt.ylabel('Gió', fontsize=14)  # Enhanced label font size
plt.colorbar(label='Lớp Thời Tiết')
plt.grid(True)  # Add grid for better readability

plt.tight_layout()
plt.show()
plt.close()

# Bước 9: Đồ thị phân tán đã chuẩn hóa
plt.figure(figsize=(12, 6))  # Adjust the figure size

# Đồ thị phân tán dữ liệu đã chuẩn hóa cho Nhiệt độ Tối Đa và Nhiệt độ Tối Thiểu
plt.subplot(1, 2, 1)  # Chỉ cần 2 cột
plt.scatter(X_data[:, 1], X_data[:, 2], c=y_data, cmap='viridis', alpha=0.7, s=50)  # Increased marker size
plt.title('Đã Chuẩn Hóa: Nhiệt độ Max vs Nhiệt độ Tối Thiểu', fontsize=16)  # Enhanced title font size
plt.xlabel('Nhiệt độ Tối Đa (Chuẩn Hóa)', fontsize=14)  # Enhanced label font size
plt.ylabel('Nhiệt độ Tối Thiểu (Chuẩn Hóa)', fontsize=14)  # Enhanced label font size
plt.colorbar(label='Lớp Thời Tiết')
plt.grid(True)  # Add grid for better readability

# Đồ thị phân tán dữ liệu đã chuẩn hóa cho Lượng Mưa và Gió
plt.subplot(1, 2, 2)  # Chỉ cần 2 cột
plt.scatter(X_data[:, 0], X_data[:, 3], c=y_data, cmap='viridis', alpha=0.7, s=50)  # Increased marker size
plt.title('Đã Chuẩn Hóa: Lượng Mưa vs Gió', fontsize=16)  # Enhanced title font size
plt.xlabel('Lượng Mưa (Chuẩn Hóa)', fontsize=14)  # Enhanced label font size
plt.ylabel('Gió (Chuẩn Hóa)', fontsize=14)  # Enhanced label font size
plt.colorbar(label='Lớp Thời Tiết')
plt.grid(True)  # Add grid for better readability

plt.tight_layout()
plt.show()
plt.close()

# Bước 10: Huấn luyện mô hình Neural Network sau khi xử lý 
# Thí nghiệm với một số cấu hình khác nhau
hidden_layer_sizes_options = [(10,), (20,), (30, 15), (40, 20, 10)]  # Các cấu hình lớp ẩn
results = []

for hidden_layer_sizes in hidden_layer_sizes_options:
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=500, activation='relu', solver='adam',
                        random_state=42, early_stopping=True, learning_rate_init=0.01)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    mean_cv_score = round(np.mean(cv_scores), 4)
    results.append((hidden_layer_sizes, mean_cv_score))
    print(f"Cấu hình lớp ẩn: {hidden_layer_sizes}, Điểm cross-validation: {mean_cv_score}")

# Bước 11: Huấn luyện mô hình với cấu hình tốt nhất
best_hidden_layer_sizes = max(results, key=lambda x: x[1])[0]
print(f"Cấu hình lớp ẩn tốt nhất: {best_hidden_layer_sizes}")

best_clf = MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes, max_iter=500, activation='relu', solver='adam',
                          random_state=42, early_stopping=True, learning_rate_init=0.01)
best_clf.fit(X_train, y_train)

# Bước 12: Dự đoán và đánh giá mô hình sau khi xử lý
y_train_pred_after = best_clf.predict(X_train)
y_val_pred_after = best_clf.predict(X_val)
y_test_pred_after = best_clf.predict(X_test)

evaluate_model(y_train, y_train_pred_after, "tập huấn luyện (sau khi xử lý)")
evaluate_model(y_val, y_val_pred_after, "tập xác thực (sau khi xử lý)")
evaluate_model(y_test, y_test_pred_after, "tập kiểm tra (sau khi xử lý)")

# Bước 13: Cross-validation sau khi tối ưu hóa
cv_scores_after = cross_val_score(best_clf, X_train, y_train, cv=5)
print(f"Cross-validation scores (after processing): {cv_scores_after}")
print(f"Mean cross-validation score (after processing): {np.mean(cv_scores_after):.4f}")
