import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu và xử lý
df = pd.read_csv('./seattle-weather.csv')
df = df.dropna()

le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])
X_data = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
y_data = df['weather_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, shuffle=True, random_state=42)

# Huấn luyện mô hình
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu', solver='adam', random_state=42)
clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)

# Hàm để hiển thị báo cáo chi tiết
def show_metrics():
    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Tạo cửa sổ mới để hiển thị kết quả
    result_window = tk.Toplevel(root)
    result_window.title("Kết quả mô hình")
    
    # Sử dụng Frame để bố trí giao diện đẹp hơn
    frame_top = tk.Frame(result_window)
    frame_top.pack(pady=10)

    # Hiển thị tỷ lệ chính xác
    label_accuracy = tk.Label(frame_top, text=f"Tỷ lệ dự đoán đúng (accuracy): {accuracy:.2f}")
    label_accuracy.grid(row=0, column=0, sticky="w")

    # Hiển thị báo cáo phân loại trong Treeview
    tree = ttk.Treeview(frame_top, columns=("precision", "recall", "f1-score", "support"), show="headings", height=5)
    tree.grid(row=2, column=0, padx=10, pady=5)

    tree.heading("precision", text="precision")
    tree.heading("recall", text="recall")
    tree.heading("f1-score", text="f1-score")
    tree.heading("support", text="support")

    # Thêm dữ liệu vào bảng Treeview
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            tree.insert("", "end", values=(f"{metrics['precision']:.2f}", f"{metrics['recall']:.2f}", f"{metrics['f1-score']:.2f}", int(metrics['support'])))

    # Tạo khung thứ hai cho các biểu đồ
    frame_bottom = tk.Frame(result_window)
    frame_bottom.pack(pady=10)

    # Tạo layout với 2 cột: một cột cho ma trận nhầm lẫn và một cột cho biểu đồ loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Hiển thị ma trận nhầm lẫn ở cột đầu tiên
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=le.classes_, yticklabels=le.classes_)
    ax1.set_title('Ma trận nhầm lẫn (Confusion Matrix)')
    ax1.set_xlabel('Nhãn dự đoán')
    ax1.set_ylabel('Nhãn thực tế')

    # Vẽ biểu đồ Loss Function ở cột thứ hai
    ax2.plot(clf.loss_curve_)
    ax2.set_title("Đường cong Loss Function")
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Loss')

    # Tạo canvas để hiển thị hình ảnh trong Tkinter
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=frame_bottom)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Hàm dự đoán thời tiết dựa trên thông tin đầu vào từ người dùng
def make_prediction():
    # Lấy giá trị từ các ô nhập liệu
    precipitation = float(entry_precipitation.get())
    temp_max = float(entry_temp_max.get())
    temp_min = float(entry_temp_min.get())
    wind = float(entry_wind.get())
    
    # Chuẩn hóa dữ liệu đầu vào
    input_data = np.array([[precipitation, temp_max, temp_min, wind]])
    input_data = scaler.transform(input_data)
    
    # Dự đoán nhãn thời tiết
    prediction = clf.predict(input_data)
    
    # Giải mã nhãn dự đoán thành tên thời tiết
    predicted_weather = le.inverse_transform(prediction)[0]
    
    # Hiển thị kết quả dự đoán
    result_label.config(text=f"Dự đoán thời tiết: {predicted_weather}")

# Xây dựng giao diện tkinter
root = tk.Tk()
root.title("Dự đoán thời tiết")

# Tạo các label và entry cho thông tin thời tiết
label_precipitation = tk.Label(root, text="Precipitation (Lượng mưa):")
label_precipitation.grid(row=0, column=0, padx=10, pady=5)

entry_precipitation = tk.Entry(root)
entry_precipitation.grid(row=0, column=1, padx=10, pady=5)

label_temp_max = tk.Label(root, text="Nhiệt độ cao nhất:")
label_temp_max.grid(row=1, column=0, padx=10, pady=5)

entry_temp_max = tk.Entry(root)
entry_temp_max.grid(row=1, column=1, padx=10, pady=5)

label_temp_min = tk.Label(root, text="Nhiệt độ thấp nhất:")
label_temp_min.grid(row=2, column=0, padx=10, pady=5)

entry_temp_min = tk.Entry(root)
entry_temp_min.grid(row=2, column=1, padx=10, pady=5)

label_wind = tk.Label(root, text="Tốc độ gió:")
label_wind.grid(row=3, column=0, padx=10, pady=5)

entry_wind = tk.Entry(root)
entry_wind.grid(row=3, column=1, padx=10, pady=5)

# Nút dự đoán
predict_button = tk.Button(root, text="Dự đoán", command=make_prediction)
predict_button.grid(row=4, columnspan=2, pady=10)

# Thêm một label để hiển thị kết quả dự đoán
result_label = tk.Label(root, text="Dự đoán thời tiết: ")
result_label.grid(row=5, columnspan=2, pady=10)

# Nút để hiển thị báo cáo chi tiết của mô hình
metrics_button = tk.Button(root, text="Hiển thị kết quả mô hình", command=show_metrics)
metrics_button.grid(row=6, columnspan=2, pady=10)

# Chạy ứng dụng
root.mainloop()
