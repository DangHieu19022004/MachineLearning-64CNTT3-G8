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



# # Plotting function for boxplots
# def plot_boxplot(df, columns):
#     plt.figure(figsize=(12, 6))
#     for i, column in enumerate(columns, 1):
#         plt.subplot(1, len(columns), i)  # Create subplots
#         sns.boxplot(y=df[column])
#         plt.title(f'Boxplot for {column}')
#     plt.tight_layout()
#     plt.show()
# columns_to_check = ['precipitation', 'temp_max', 'temp_min', 'wind']
# # Call the function with the appropriate columns
# plot_boxplot(df, columns_to_check)

print("mô tả")
print(df.describe())

# print("5 hang DL dau tien")
# print(df.head())

# print("thông tin DL")
# print(df.info())

# #truoc khi chuan hoa
# print("mô tả")
# print(df.describe())

# #ma trận tương quan
# print("ma trận tương quan trc khi chuan hoa")
# sns.heatmap(X_before.corr(), annot=True)
# plt.show()  # Hiển thị biểu đồ

# # Tạo một biểu đồ phân phối cho từng biến trong X trc khi chuẩn hóa
# plt.figure(figsize=(15, 10))
# # Vẽ phân phối cho từng cột trong X trc khi chuẩn hóa
# for i, column in enumerate(X_before.columns):
#     plt.subplot(2, 2, i + 1)  # Tạo các subplot
#     sns.histplot(X_before[column], kde=True, bins=30)  # Sử dụng histplot với đường KDE
#     plt.title(f'Distribution of {column}')  # Tiêu đề cho từng biểu đồ
#     plt.xlabel(column)  # Nhãn cho trục x
#     plt.ylabel('Frequency')  # Nhãn cho trục y

# plt.tight_layout()  # Căn chỉnh các biểu đồ
# plt.show()  # Hiển thị biểu đồ

# #tính trung bình trc khi chuan hoa
# print("trung bình")
# print(X_before.mean())
# print("trung vị - giá trị ở giữa")
# print(X_before.median())
# print("giá trị nhiều nhất")
# print(df.mode())


# # Chuan hoa dl
# # Kiểm tra dữ liệu thiếu
# df.isnull().sum()
# # Loại bỏ các hàng có giá trị thiếu
# df.dropna(inplace=True)
# # # Mã hóa cột weather thành số
# # df['weather'] = df['weather'].astype('category').cat.codes
# # Bỏ cột date
# df.drop(['date'], axis=1, inplace=True)

# print("mô tả sau")
# print(df.describe())

# #sau chuan hoa
# X_after = df[['precipitation', 'temp_max', 'temp_min', 'wind']]  # đặc trưng
# y_after = df['weather']  # nhãn mục tiêu

# #ma trận tương quan
# print("ma trận tương quan sau khi chuan hoa")
# sns.heatmap(X_after.corr(), annot=True)

# # Tạo một biểu đồ phân phối cho từng biến trong X sau khi chuẩn hóa
# plt.figure(figsize=(15, 10))

# # Vẽ phân phối cho từng cột trong X sau khi chuẩn hóa
# for i, column in enumerate(X_after.columns):
#     plt.subplot(2, 2, i + 1)  # Tạo các subplot
#     sns.histplot(X_after[column], kde=True, bins=30)  # Sử dụng histplot với đường KDE
#     plt.title(f'Distribution of {column}')  # Tiêu đề cho từng biểu đồ
#     plt.xlabel(column)  # Nhãn cho trục x
#     plt.ylabel('Frequency')  # Nhãn cho trục y

# plt.tight_layout()  # Căn chỉnh các biểu đồ
# plt.show()  # Hiển thị biểu đồ

# #tính trung bình sau khi chuan hoa
# print("trung bình")
# print(X_after.mean())
# print("trung vị - giá trị ở giữa")
# print(X_after.median())
# print("giá trị nhiều nhất")
# print(df.mode())
