import base64
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    ('decision_tree', tree.DecisionTreeClassifier(criterion='entropy', random_state=42,
                                    class_weight = None,
                                    max_depth = 3,
                                    max_features = 'sqrt',
                                    max_leaf_nodes = None,
                                    min_samples_leaf = 1
                                    , min_samples_split = 2,
                                    min_weight_fraction_leaf = 0.0,
                                    splitter = 'best')),
    ('perceptron', Perceptron(max_iter=1000, eta0=0.01, penalty='l2', alpha=0, tol=1e-4, random_state=42)),
    ('neural_network', MLPClassifier(hidden_layer_sizes=(40, 20, 10),
                                max_iter=500,
                                activation='tanh',
                                solver='adam',
                                random_state=42,
                                early_stopping=True,
                                learning_rate_init=0.01))
]

# Khởi tạo mô hình Stacking
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=MLPClassifier(hidden_layer_sizes=(5,), alpha=0.01, max_iter=1000, random_state=42))

# Huấn luyện mô hình Stacking
stacking_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_test_pred = stacking_model.predict(X_test)

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy:.2f}")

# Hiển thị báo cáo phân loại
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# Hiển thị ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.show()

# Hàm vẽ learning curve
def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    
    # Tính trung bình và độ lệch chuẩn của các điểm dữ liệu
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    
    # Vẽ vùng độ lệch chuẩn
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # Vẽ đường trung bình
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Vẽ learning curve cho mô hình Stacking
plot_learning_curve(stacking_model, X_train, y_train, title="Learning Curve for Stacking Model")


valueSend = {
    'model': stacking_model,
}

joblib.dump(valueSend, 'stacking_model.pkl')