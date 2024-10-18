import base64
import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.linear_model import Perceptron  # Sử dụng Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, learning_curve,
                                     train_test_split)
from sklearn.preprocessing import label_binarize

# get the dataset
data = pd.read_csv('./seattle-weather.csv')

# filter data
data = data.dropna()
data.drop(['date'], axis=1, inplace=True)

# Split the datasets into X and y
X = data[["precipitation", "temp_max", "temp_min", "wind"]]
y = data["weather"]

# Split the datasets into 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Khởi tạo mô hình Perceptron
perceptron_model = Perceptron(max_iter=1000, random_state=42)

# Define the parameter grid for cross-validation
param_grid = {
    'max_iter': [100, 200, 500, 1000],
    'tol': [1e-3, 1e-4, 1e-5],
    'eta0': [1.0, 0.1, 0.01],
    'penalty': [None, 'l2', 'l1', 'elasticnet'], 
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],                               
    'validation_fraction': [0.1, 0.15, 0.2]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=perceptron_model, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

# Train with cross-validation
grid_search.fit(X_train, y_train)

# Output the best parameters found
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")

# Evaluate on the validation set
best_model = grid_search.best_estimator_

# Huấn luyện mô hình với các tham số tốt nhất
best_model.fit(X_train, y_train)

# Dự đoán
y_train_pred = best_model.predict(X_train)
y_valid_pred = best_model.predict(X_valid)
y_test_pred = best_model.predict(X_test)

# Tính toán độ chính xác mô hình
accuracy = accuracy_score(y_test, y_test_pred) * 100
print(f"Test accuracy: {accuracy:.2f}%")

# Lưu ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix on test set")

# In báo cáo phân loại (precision, recall, F1-score cho từng lớp)
# 3.1. report training set
report_training_set = classification_report(y_train, y_train_pred, target_names=best_model.classes_, zero_division=0)
# 3.2. report validation set
report_validation = classification_report(y_valid, y_valid_pred, target_names=best_model.classes_, zero_division=0)
# 3.3. report test set
report_test_set = classification_report(y_test, y_test_pred, target_names=best_model.classes_, zero_division=0)

print("Training Report:")
print(report_training_set)
print("Validation Report:")
print(report_validation)
print("Test Report:")
print(report_test_set)

# Vẽ sơ đồ learning curve cho cả tập huấn luyện, xác thực và kiểm tra
train_sizes, train_scores, valid_scores = learning_curve(
    best_model, X_train, y_train, cv=kf, scoring='accuracy',
    n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
)

# Tính toán điểm trung bình và khoảng tin cậy
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

# Tính điểm accuracy cho tập test ở từng kích thước của train_sizes
test_scores = []
for train_size in train_sizes:
    # Tạo tập con của dữ liệu huấn luyện
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
    best_model.fit(X_train_subset, y_train_subset)
    test_scores.append(accuracy_score(y_test, best_model.predict(X_test)))

# Chuyển đổi test_scores sang numpy array
test_scores = np.array(test_scores)

# Vẽ learning curve
plt.figure()
plt.title('Learning Curve (Perceptron)')
plt.xlabel('Training examples')
plt.ylabel('Score')

# Trực quan hóa khoảng tin cậy của tập huấn luyện và xác thực
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.1, color="g")

# Vẽ các đường learning curve
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Validation score")
plt.plot(train_sizes, test_scores, 'o-', color="b", label="Test score")

# Hiển thị legend và biểu đồ
plt.legend(loc="best")
plt.show()

valueSend = {
    'model': perceptron_model,
}

joblib.dump(valueSend, 'perceptron_model.pkl')
