import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
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

# Bước 5: Định nghĩa mạng nơ-ron và tham số cần tìm kiếm
param_grid_nn = {
    'hidden_layer_sizes': [
        (10,), (20,), (30,),  
        (20, 10), (30, 20, 10), 
        (40, 20, 10), (50, 30, 10)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate_init': [0.1, 0.01, 0.001, 0.0001],
    'max_iter': [500, 1000],
    'early_stopping': [True]
}

# Khởi tạo MLPClassifier và GridSearchCV
mlp_model = MLPClassifier(random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_nn = GridSearchCV(estimator=mlp_model, param_grid=param_grid_nn, cv=kf, scoring='accuracy', n_jobs=-1)

# Bước 6: Huấn luyện mô hình với cross-validation
grid_search_nn.fit(X_train, y_train)

# Xuất các tham số tốt nhất 
print(f"Best parameters found for Neural Network: {grid_search_nn.best_params_}")
print(f"Best cross-validation accuracy for Neural Network: {grid_search_nn.best_score_:.4f}")

# Bước 7: Đánh giá mô hình với tham số tốt nhất trên tập validation và test
best_nn_model = grid_search_nn.best_estimator_
y_train_pred_nn = best_nn_model.predict(X_train)
y_val_pred_nn = best_nn_model.predict(X_val)
y_test_pred_nn = best_nn_model.predict(X_test)

# Đánh giá độ chính xác
train_accuracy = round(accuracy_score(y_train, y_train_pred_nn), 4)
val_accuracy = round(accuracy_score(y_val, y_val_pred_nn), 4)
test_accuracy = round(accuracy_score(y_test, y_test_pred_nn), 4)

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# In báo cáo phân loại cho từng tập dữ liệu
print("\nClassification Report on Training Set")
print(classification_report(y_train, y_train_pred_nn, target_names=le.classes_, zero_division=0))

print("\nClassification Report on Validation Set")
print(classification_report(y_val, y_val_pred_nn, target_names=le.classes_, zero_division=0))

print("\nClassification Report on Test Set")
print(classification_report(y_test, y_test_pred_nn, target_names=le.classes_, zero_division=0))

# Lưu mô hình đã huấn luyện và dữ liệu cần thiết
joblib.dump(best_nn_model, 'best_nn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump((X_train, X_test, y_train, y_test), 'data_splits.pkl')