import base64
import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu
data = pd.read_csv("./seattle-weather.csv")

# Lọc dữ liệu
data = data.dropna()

# Các cột đặc trưng để huấn luyện mô hình
features = ["precipitation", "temp_max", "temp_min", "wind"]

# Chia dữ liệu thành X và y
X = data[features]
y = data["weather"]

# Mã hóa nhãn
le = LabelEncoder()
y = le.fit_transform(y)

# Chia dữ liệu thành 70% huấn luyện, 15% xác thực, 15% kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Khởi tạo và huấn luyện mô hình Perceptron
perceptron_model = Perceptron(random_state=42)
perceptron_model.fit(X_train, y_train)

# Dự đoán trên các tập dữ liệu
y_train_pred = perceptron_model.predict(X_train)
y_valid_pred = perceptron_model.predict(X_valid)
y_test_pred = perceptron_model.predict(X_test)

# Báo cáo phân loại
report_validation = classification_report(y_valid, y_valid_pred, target_names=le.classes_, zero_division=0)
report_trainning_set = classification_report(y_train, y_train_pred, target_names=le.classes_, zero_division=0)
report_test_set = classification_report(y_test, y_test_pred, target_names=le.classes_, zero_division=0)

print("Báo cáo cho tập xác thực:")
print(report_validation)
print("Báo cáo cho tập huấn luyện:")
print(report_trainning_set)
print("Báo cáo cho tập kiểm tra:")
print(report_test_set)

# Ma trận nhầm lẫn cho tập kiểm tra
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix")
img = io.BytesIO()
plt.savefig(img, format='png')
img.seek(0)
plot_url = base64.b64encode(img.getvalue()).decode()
# plt.show()

# Learning curve
train_sizes, train_scores, valid_scores = learning_curve(
    perceptron_model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Validation score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid(True)
imgLearningCurve = io.BytesIO()
plt.savefig(imgLearningCurve, format='png')
imgLearningCurve.seek(0)
learning_curve_url = base64.b64encode(imgLearningCurve.getvalue()).decode()
# plt.show()

# Lưu mô hình và kết quả
valueSend = {
    'model': perceptron_model,
    'report_validation': report_validation,
    'report_trainning_set': report_trainning_set,
    'report_test_set': report_test_set,
    'plot_url': plot_url,
    'learning_curve_url': learning_curve_url
}

joblib.dump(valueSend, 'perceptron_model.pkl')



# Example: Predict new data
# Create a DataFrame for new data (example values, replace with actual data as needed)
# new_data = pd.DataFrame({
#     "precipitation": [0.1, 0.2, 0.0],   # Example values
#     "temp_max": [18.0, 22.0, 20.0],
#     "temp_min": [10.0, 12.0, 11.0],
#     "wind": [5.0, 7.0, 6.0]
# })

# # Predict using the trained Perceptron model
# new_data_predictions = perceptron_model.predict(new_data)

# # Print the predictions
# print("Predictions for new data:")
# for i, prediction in enumerate(new_data_predictions):
#     print(f"Data point {i + 1}: {le.inverse_transform([prediction])[0]}")

