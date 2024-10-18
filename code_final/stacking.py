import base64
import io
import joblib
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

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
    ('decision_tree', tree.DecisionTreeClassifier(max_depth=5, random_state=42)),  # Tăng độ sâu một chút
    ('perceptron', Perceptron(max_iter=1000, random_state=42)),
    ('neural_network', MLPClassifier(hidden_layer_sizes=(5,), alpha=0.01, max_iter=2000, random_state=42))  # Giảm số lượng nút trong lớp ẩn và thêm regularization
]

# Khởi tạo mô hình Stacking
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=MLPClassifier(hidden_layer_sizes=(5,), alpha=0.01, max_iter=1000, random_state=42))  

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

# Vẽ Learning Curve
train_sizes, train_scores, valid_scores = learning_curve(stacking_model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Tính toán điểm trung bình và độ lệch chuẩn cho training scores và validation scores
train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

# Vẽ biểu đồ Learning Curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, valid_scores_mean, 'o-', color='g', label='Validation score')

# Vẽ độ lệch chuẩn
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')

plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.show()

valueSend = {
    'model': stacking_model,
}

joblib.dump(valueSend, 'stacking_model.pkl')