import base64
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

#1. model
#2. report validation
#3. report training set
#4. report test set
#5. ma trận nhầm lẫn (plot_url)
#6. ma trận learning curve

# Đọc dữ liệu từ file CSV
df = pd.read_csv("./seattle-weather.csv")

# Xử lý dữ liệu thiếu (nếu có) bằng cách loại bỏ các hàng có giá trị null
df = df.dropna()

# Mã hóa thuộc tính weather (categorical) thành số
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])

# Sử dụng các cột numerical như precipitation, temp_max, temp_min, wind làm dữ liệu đầu vào
X_data = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values

# Chuẩn hóa (scale) các thuộc tính numerical để phù hợp với MLP
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

# Nhãn (target) là cột weather đã được mã hóa
y_data = df['weather_encoded'].values

# Chia tập dữ liệu thành 3 tập: training (70%), validation (15%), test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.3, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

# Xây dựng mô hình MLP (Multilayer Perceptron - Neural Network)
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu', solver='adam', random_state=42)

# Huấn luyện mô hình
clf.fit(X_train, y_train)

# Dự đoán trên các tập dữ liệu
y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

# Đánh giá mô hình trên cả 3 tập
train_report = classification_report(y_train, y_train_pred, target_names=le.classes_, zero_division=1)
val_report = classification_report(y_val, y_val_pred, target_names=le.classes_, zero_division=1)
test_report = classification_report(y_test, y_test_pred, target_names=le.classes_, zero_division=1)

def encode_input(new_input):
    predicted_encoded = clf.predict(new_input)
    predicted_labels = le.inverse_transform(predicted_encoded)
    return predicted_labels

# Vẽ ma trận nhầm lẫn (Confusion Matrix) cho tập validation
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)

    # Lưu ma trận nhầm lẫn vào bộ nhớ dưới dạng ảnh base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

confusion_matrix_base64 = plot_confusion_matrix(y_val, y_val_pred, 'Confusion Matrix - Validation Set')

# Vẽ biểu đồ Learning Curve
def plot_learning_curve(estimator, X, y, title, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")

    # Lấy giá trị training và validation scores cho từng kích thước tập huấn luyện
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    # Tính trung bình và độ lệch chuẩn
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Vẽ biểu đồ training và validation scores
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Validation score")

    plt.legend(loc="best")

    # Lưu Learning Curve vào bộ nhớ dưới dạng ảnh base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# Lấy ảnh base64 của Learning Curve
learning_curve_base64 = plot_learning_curve(clf, X_train, y_train, "Learning Curve - MLP Classifier", cv=5)

valueSend = {
    'model': clf,
    'report_validation': val_report,
    'report_trainning_set': train_report,
    'report_test_set': test_report,
    'plot_url': confusion_matrix_base64,
    'learning_curve_url': learning_curve_base64
}


# Save the model
joblib.dump(valueSend, 'neural_network_model.pkl')
