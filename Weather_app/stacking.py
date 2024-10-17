import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier  # Nhập MLPClassifier để sử dụng mạng nơ-ron
import matplotlib.pyplot as plt  # Nhập matplotlib để hiển thị biểu đồ

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

# Khởi tạo các mô hình cơ bản
base_learners = [
    ('decision_tree', tree.DecisionTreeClassifier(max_depth=3, random_state=42)),
    ('perceptron', Perceptron(max_iter=1000, random_state=42)),
    ('neural_network', MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=42))  
]

# Khởi tạo mô hình Stacking
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42))  

# Huấn luyện mô hình Stacking
stacking_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_test_pred = stacking_model.predict(X_test)

# Tính toán độ chính xác mô hình
accuracy = accuracy_score(y_test, y_test_pred) * 100
print(f"Test accuracy: {accuracy:.2f}%")

# In báo cáo phân loại
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_, zero_division=0))

# Lưu ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix on test set")
plt.show()

