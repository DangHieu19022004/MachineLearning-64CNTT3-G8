#import libraries
import base64
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn import tree
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("./seattle-weather.csv")

# Tách đặc trưng và nhãn
X_before = df[['precipitation', 'temp_max', 'temp_min', 'wind']]  # đặc trưng
y_before = df['weather']  # nhãn mục tiêu

#truoc khi chuan hoa
print("mô tả")
print(df.describe())

#ma trận tương quan
print("ma trận tương quan trc khi chuan hoa")
sns.heatmap(X_before.corr(), annot=True)

# Tạo một biểu đồ phân phối cho từng biến trong X trc khi chuẩn hóa
plt.figure(figsize=(15, 10))
# Vẽ phân phối cho từng cột trong X trc khi chuẩn hóa
for i, column in enumerate(X_before.columns):
    plt.subplot(2, 2, i + 1)  # Tạo các subplot
    sns.histplot(X_before[column], kde=True, bins=30)  # Sử dụng histplot với đường KDE
    plt.title(f'Distribution of {column}')  # Tiêu đề cho từng biểu đồ
    plt.xlabel(column)  # Nhãn cho trục x
    plt.ylabel('Frequency')  # Nhãn cho trục y

plt.tight_layout()  # Căn chỉnh các biểu đồ
plt.show()  # Hiển thị biểu đồ

#tính trung bình trc khi chuan hoa
print("trung bình")
print(X_before.mean())
print("trung vị - giá trị ở giữa")
print(X_before.median())
print("giá trị nhiều nhất")
print(df.mode())


# Chuan hoa dl
# Kiểm tra dữ liệu thiếu
df.isnull().sum()
# Loại bỏ các hàng có giá trị thiếu
df.dropna(inplace=True)
# # Mã hóa cột weather thành số
# df['weather'] = df['weather'].astype('category').cat.codes
# Bỏ cột date
df.drop(['date'], axis=1, inplace=True)

print("mô tả sau")
print(df.describe())

#sau chuan hoa
X_after = df[['precipitation', 'temp_max', 'temp_min', 'wind']]  # đặc trưng
y_after = df['weather']  # nhãn mục tiêu

#ma trận tương quan
print("ma trận tương quan sau khi chuan hoa")
sns.heatmap(X_after.corr(), annot=True)

# Tạo một biểu đồ phân phối cho từng biến trong X sau khi chuẩn hóa
plt.figure(figsize=(15, 10))

# Vẽ phân phối cho từng cột trong X sau khi chuẩn hóa
for i, column in enumerate(X_after.columns):
    plt.subplot(2, 2, i + 1)  # Tạo các subplot
    sns.histplot(X_after[column], kde=True, bins=30)  # Sử dụng histplot với đường KDE
    plt.title(f'Distribution of {column}')  # Tiêu đề cho từng biểu đồ
    plt.xlabel(column)  # Nhãn cho trục x
    plt.ylabel('Frequency')  # Nhãn cho trục y

plt.tight_layout()  # Căn chỉnh các biểu đồ
plt.show()  # Hiển thị biểu đồ

#tính trung bình sau khi chuan hoa
print("trung bình")
print(X_after.mean())
print("trung vị - giá trị ở giữa")
print(X_after.median())
print("giá trị nhiều nhất")
print(df.mode())


# #trainning model
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
# X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # Khởi tạo scaler
# scaler = StandardScaler()

# # Chuẩn hóa tập huấn luyện (fit_transform chỉ trên tập huấn luyện)
# X_train_scaled = scaler.fit_transform(X_train)

# # Chuẩn hóa tập validation và test (chỉ transform, không fit lại)
# X_valid_scaled = scaler.transform(X_valid)
# X_test_scaled = scaler.transform(X_test)

# #trainning the model
# dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42, min_samples_split=5, min_samples_leaf=3)

# #fit trainnign data into model
# dt_model.fit(X_train_scaled, y_train)

# #predict the model on test data
# y_train_pred = dt_model.predict(X_train_scaled)
# y_valid_pred = dt_model.predict(X_valid_scaled)
# y_test_pred = dt_model.predict(X_test_scaled)

# print(df['weather'].value_counts())


# #vẽ roc và aug
# from sklearn.metrics import auc, roc_curve

# # Tính toán TPR và FPR
# y_prob = dt_model.predict_proba(X_valid_scaled)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_valid, y_prob, pos_label=1)
# roc_auc = auc(fpr, tpr)

# # Vẽ biểu đồ ROC
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()



# # In báo cáo phân loại (precision, recall, F1-score cho từng lớp)
# # 3.1. report training set
# report_trainning_set = classification_report(y_train, y_train_pred,  target_names=dt_model.classes_, zero_division=0)
# # 3.2. report validation set
# report_validation = classification_report(y_valid, y_valid_pred, target_names=dt_model.classes_, zero_division=0)
# # 3.3. report test set
# report_test_set = classification_report(y_test, y_test_pred, target_names=dt_model.classes_, zero_division=0)

# print("trainning")
# print(report_trainning_set)
# print("vali")
# print(report_validation)
# print("test")
# print(report_test_set)

# # 2. Lưu ma trận nhầm lẫn
# cm = confusion_matrix(y_test, y_test_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
# fig, ax = plt.subplots(figsize=(10, 7))
# disp.plot(cmap=plt.cm.Blues, ax=ax)
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.title("Confusion Matrix on test set")
# plt.show()

# # 4. Vẽ sơ đồ learning curve
# train_sizes, train_scores, valid_scores = learning_curve(dt_model,  X_train_scaled, y_train,  train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy')
# # Tính giá trị trung bình và độ lệch chuẩn của các điểm số
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# valid_scores_mean = np.mean(valid_scores, axis=1)
# valid_scores_std = np.std(valid_scores, axis=1)
# # Vẽ biểu đồ learning curve
# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Validation score")
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
#                  valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
# plt.title("Learning Curve")
# plt.xlabel("Training Examples")
# plt.ylabel("Score")
# plt.legend(loc="best")
# plt.grid(True)
# plt.show()

# #####
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import log_loss
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier

# train_sizes = np.linspace(0.1, 0.9, 10)  # Thay đổi kích thước tập huấn luyện
# train_entropy = []
# valid_entropy = []

# for size in train_sizes:
#     X_train_subset, X_valid_subset, y_train_subset, y_valid_subset = train_test_split(X, y, train_size=size, random_state=42)

#     # Huấn luyện mô hình
#     dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42, min_samples_split=5, min_samples_leaf=3)

#     # Khởi tạo scaler
#     scaler = StandardScaler()
#     X_train_scaled_subset = scaler.fit_transform(X_train_subset)
#     X_valid_scaled_subset = scaler.transform(X_valid_subset)

#     dt_model.fit(X_train_scaled_subset, y_train_subset)

#     # Dự đoán trên tập huấn luyện và tập validation
#     y_train_probs = dt_model.predict_proba(X_train_scaled_subset)
#     y_valid_probs = dt_model.predict_proba(X_valid_scaled_subset)

#     # Tính toán log loss (entropy)
#     train_entropy.append(log_loss(y_train_subset, y_train_probs))
#     valid_entropy.append(log_loss(y_valid_subset, y_valid_probs))

# # Vẽ biểu đồ
# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_entropy, 'o-', color="r", label="Training Entropy")
# plt.plot(train_sizes, valid_entropy, 'o-', color="g", label="Validation Entropy")
# plt.title("Entropy (Log Loss) Curve")
# plt.xlabel("Training Examples")
# plt.ylabel("Entropy (Log Loss)")
# plt.legend(loc="best")
# plt.grid(True)
# plt.show()
