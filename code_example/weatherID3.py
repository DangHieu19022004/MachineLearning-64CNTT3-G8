#predict weather based on dataset

#import libraries
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
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)

#fit trainnign data into model
dt_model.fit(X_train, y_train)

# # Save the model
# joblib.dump(dt_model, 'weather_model.pkl')

#predict the model on test data
y_preds = dt_model.predict(X_valid)

#predict with new input
new_input = pd.DataFrame({
    'precipitation': [4.3],
    'temp_max': [13.9],
    'temp_min': [10],
    'wind': [2.8]
})

print(dt_model.predict(new_input))

#Pain the model graphics
# compare the prediction with the actual values
cm = confusion_matrix(y_valid, y_preds)

print(cm)

# # Display confusion matrix in graphical view
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
# disp.plot()
# plt.show()
# # display the tree in graphical view
# # tree.plot_tree(dt_model, feature_names = ['precipitation', 'temp_max', 'temp_min', 'wind'], class_names = dt_model.classes_)
# # plt.show()


#Model Evaluation
# 1. Tính toán độ chính xác của mô hình
accuracy = accuracy_score(y_valid, y_preds)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 2. In ra ma trận nhầm lẫn
cm = confusion_matrix(y_valid, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 3. In báo cáo phân loại (precision, recall, F1-score cho từng lớp)
report = classification_report(y_valid, y_preds, target_names=dt_model.classes_)
print("Classification Report:\n", report)
