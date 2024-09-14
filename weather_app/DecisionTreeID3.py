#predict weather based on dataset

#import libraries
import base64
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#get the dataset
data = pd.read_csv("../seattle-weather.csv", index_col=0)

#features columns to train the model
features = ["precipitation", "temp_max", "temp_min", "wind"]

#Split the datasets into X and y
X = data[features]
y = data["weather"]

#Split the datasets into training and testing
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

#trainning the model
dt_model = DecisionTreeClassifier(criterion='entropy')

#fit trainnign data into model
dt_model.fit(X_train, y_train)

#predict the model on test data
y_preds = dt_model.predict(X_valid)

#Model Evaluation
# 1. Tính toán độ chính xác của mô hình
accuracy = accuracy_score(y_valid, y_preds) * 100

# 2. Lưu ma trận nhầm lẫn
cm = confusion_matrix(y_valid, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix")
    #Save the plot to a BytesIO object
img = io.BytesIO()
plt.savefig(img, format='png')
img.seek(0)
    #Encode image to base64
plot_url = base64.b64encode(img.getvalue()).decode()

# 3. In báo cáo phân loại (precision, recall, F1-score cho từng lớp)
report = classification_report(y_valid, y_preds, target_names=dt_model.classes_)

valueSend = {
    'model': dt_model,
    'accuracy': accuracy,
    'report': report,
    'plot_url': plot_url
}

# Save the model
joblib.dump(valueSend, 'weather_model.pkl')
