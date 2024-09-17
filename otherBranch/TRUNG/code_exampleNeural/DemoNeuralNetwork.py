import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

# Đọc dữ liệu từ file CSV
df = pd.read_csv('../seattle-weather.csv')

# Xử lý dữ liệu thiếu (nếu có) bằng cách loại bỏ các hàng có giá trị null
df = df.dropna()

# Mã hóa thuộc tính weather (categorical) thành số
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])

# Lưu LabelEncoder để sử dụng trong quá trình deploy lên web
joblib.dump(le, 'label_encoder.pkl')  # Lưu LabelEncoder để dùng lại sau

# Sử dụng các cột numerical như precipitation, temp_max, temp_min, wind làm dữ liệu đầu vào
X_data = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values

# Chuẩn hóa (scale) các thuộc tính numerical để phù hợp với MLP
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

# Lưu scaler để sử dụng trong quá trình deploy lên web
joblib.dump(scaler, 'scaler.pkl')  # Lưu StandardScaler để dùng lại sau

# Nhãn (target) là cột weather đã được mã hóa
y_data = df['weather_encoded'].values

# Chia tập dữ liệu thành 3 tập: training (60%), validation (20%), test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.4, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

# Show data của các tập
print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set:", X_test.shape, y_test.shape)

# Xây dựng mô hình MLP (Multilayer Perceptron - Neural Network)
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu', solver='adam', random_state=42)

# Huấn luyện mô hình
clf.fit(X_train, y_train)

# Lưu mô hình để triển khai sau này
joblib.dump(clf, 'neural_network_model.pkl')

# Dự đoán trên các tập dữ liệu
y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

# Đánh giá mô hình trên cả 3 tập
print("\nĐánh giá trên tập Training:")
print(classification_report(y_train, y_train_pred, target_names=le.classes_, zero_division=1))
print("\nĐánh giá trên tập Validation:")
print(classification_report(y_val, y_val_pred, target_names=le.classes_, zero_division=1))
print("\nĐánh giá trên tập Test:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_, zero_division=1))

# In độ chính xác (accuracy) của cả 3 tập
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# In ma trận nhầm lẫn (Confusion Matrix) cho cả 3 tập
def plot_confusion_matrix(y_true, y_pred, title):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()

# 1. Vẽ ma trận nhầm lẫn (Confusion Matrix)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('True label')  # Nhãn thực tế
plt.xlabel('Predicted label')  # Nhãn dự đoán
plt.title('Confusion Matrix')  # Tiêu đề: Ma trận nhầm lẫn
plot_confusion_matrix(y_train, y_train_pred, 'Confusion Matrix - Training Set')
plot_confusion_matrix(y_val, y_val_pred, 'Confusion Matrix - Validation Set')
plot_confusion_matrix(y_test, y_test_pred, 'Confusion Matrix - Test Set')

# 2. Vẽ biểu đồ thể hiện sự tuyến tính của dữ liệu (tạm thời dùng biểu đồ tuyến tính của dữ liệu đầu vào)
plt.figure(figsize=(8, 6))
plt.plot(X_data)
plt.title('Linearity of the Input Data')
plt.xlabel('Sample Index')
plt.ylabel('Scaled Feature Values')
plt.show()

# 3. Vẽ đồ thị hàm mất mát (Loss function) trong quá trình huấn luyện
plt.figure(figsize=(8, 6))
plt.plot(clf.loss_curve_)
plt.title('Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# In ra thứ tự mã hóa của các nhãn
print("Thứ tự mã hóa của các nhãn (LabelEncoder):")
for i, label in enumerate(le.classes_):
    print(f'{label} -> {i}')






















'''
# 2. Vẽ biểu đồ heatmap của báo cáo phân loại (Classification Report)
# Tạo báo cáo phân loại và chuyển đổi thành DataFrame (thêm zero_division=1)
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=1)
report_df = pd.DataFrame(report_dict).transpose()

# Vẽ heatmap của các chỉ số báo cáo phân loại
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report Heatmap')  # Tiêu đề: Báo cáo phân loại dưới dạng heatmap
plt.show()

# 3. Vẽ đường cong ROC cho từng lớp (ROC Curve)
# Binarize (mã hóa nhị phân) nhãn đầu ra cho phân loại đa lớp ROC
y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))
y_pred_bin = label_binarize(y_pred, classes=np.arange(len(le.classes_)))

# Vẽ đường cong ROC cho từng lớp
plt.figure(figsize=(8, 6))
for i, class_name in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])  # Tính FPR và TPR cho lớp i
    roc_auc = auc(fpr, tpr)  # Tính AUC cho lớp i
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')  # Vẽ ROC cho từng lớp

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Đường tham chiếu ROC
plt.xlim([0.0, 1.0])  # Giới hạn trục X
plt.ylim([0.0, 1.05])  # Giới hạn trục Y
plt.xlabel('False Positive Rate')  # Trục X: Tỷ lệ dương tính giả
plt.ylabel('True Positive Rate')  # Trục Y: Tỷ lệ dương tính thực
plt.title('ROC Curve for Multi-Class Classification')  # Tiêu đề: Đường cong ROC cho phân loại đa lớp
plt.legend(loc='lower right')  # Hiển thị chú thích ở góc phải dưới
plt.show()
'''
