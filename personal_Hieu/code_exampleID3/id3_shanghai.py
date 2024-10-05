import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
data = pd.read_csv("./Shanghai AQI and Wheather 2014-2021.csv")

# Lọc dữ liệu
data = data.dropna()

features = ["maxtempC", "mintempC", "totalSnow_cm", "sunHour"]
X = data[features]
y = data["AQI_Explained"].dropna()  # Lưu ý là đã xử lý NaN trong y

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["AQI_Explained"].dropna())

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình hồi quy tuyến tính
linear_model = LinearRegression()

# Huấn luyện mô hình
linear_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = linear_model.predict(X_test)

# Tính toán hàm mất mát (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Vẽ đồ thị hàm mất mát
loss_values = []
for i in np.linspace(0.1, 10, 100):
    y_pred_temp = linear_model.predict(X_test * i)  # Thay đổi đầu vào để kiểm tra ảnh hưởng đến hàm mất mát
    loss_values.append(mean_squared_error(y_test, y_pred_temp))

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0.1, 10, 100), loss_values, color="b", label="Loss Function (MSE)")
plt.title("Loss Function of Linear Regression")
plt.xlabel("Input Scaling Factor")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend(loc="best")
plt.grid(True)
plt.show()
