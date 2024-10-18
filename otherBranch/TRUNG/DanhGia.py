import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, learning_curve
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

# 1. Ma trận nhầm lẫn
cm_nn = confusion_matrix(y_test, y_test_pred_nn)
disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp_nn.plot(cmap=plt.cm.Blues, ax=ax)
plt.ylabel('Nhãn Thực')
plt.xlabel('Nhãn Dự Đoán')
plt.title("Ma trận nhầm lẫn cho Mạng Nơ-ron trên Tập Kiểm Tra")
plt.show()

# 2. Đường cong học tập (Learning Curve)
train_sizes, train_scores, val_scores = learning_curve(
    best_nn_model, X_train, y_train, cv=5, scoring='accuracy', 
    n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
)

# Tính giá trị trung bình và độ lệch chuẩn cho điểm số huấn luyện và xác thực
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Tính điểm số trên tập test cho mỗi kích thước tập huấn luyện
test_scores = []
for size in train_sizes:
    # Chia dữ liệu theo kích thước tập huấn luyện
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=size/X_train.shape[0], random_state=42)
    
    # Huấn luyện mô hình với tập con và dự đoán trên tập kiểm tra
    best_nn_model.fit(X_train_subset, y_train_subset)
    y_test_pred = best_nn_model.predict(X_test)
    
    # Đánh giá điểm số trên tập kiểm tra
    test_scores.append(accuracy_score(y_test, y_test_pred))

test_scores = np.array(test_scores)

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label="Validation score")
plt.plot(train_sizes, test_scores, 'o-', color='b', label="Test score")

# Thêm các vùng che phủ (confidence interval) cho điểm số huấn luyện và xác thực
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color='g')

# Thiết lập tiêu đề và nhãn
plt.title("Learning Curve (Neural Network)")
plt.xlabel("Training examples")
plt.ylabel("Accuracy Score")
plt.legend(loc="best")
plt.grid()
plt.show()

# Huấn luyện lại mô hình để lưu trữ loss curve
best_nn_model.fit(X_train, y_train)

# Vẽ biểu đồ hàm mất mát (Loss Curve)
plt.figure(figsize=(10, 6))
plt.plot(best_nn_model.loss_curve_, label='Training Loss', color='purple')
plt.title("Loss Curve for Neural Network")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.grid()
plt.show()