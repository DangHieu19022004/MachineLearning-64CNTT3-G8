#predict weather based on dataset

#import libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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

#Model Evaluation
# compare the prediction with the actual values
cm = confusion_matrix(y_valid, y_preds)

print(cm)

# Display confusion matrix in graphical view
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
disp.plot()
plt.show()
# display the tree in graphical view
# tree.plot_tree(dt_model, feature_names = ['precipitation', 'temp_max', 'temp_min', 'wind'], class_names = dt_model.classes_)
# plt.show()
