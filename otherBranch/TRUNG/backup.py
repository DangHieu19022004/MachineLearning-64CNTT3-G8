import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./seattle-weather.csv')

# Xử lý dữ liệu thiếu
df = df.dropna()

# Mã hóa cột weather thành dạng số bằng LabelEncoder
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])

# Lưu LabelEncoder để sử dụng trong quá trình deploy lên web
joblib.dump(le, 'label_encoder.pkl')  # Lưu LabelEncoder để dùng lại sau

# Sử dụng các cột numerical như precipitation, temp_max, temp_min, wind làm dữ liệu đầu vào
X_data = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values


X_data = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values
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

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)
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