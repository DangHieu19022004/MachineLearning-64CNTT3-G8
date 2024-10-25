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
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier

#get the dataset
data = pd.read_csv("./seattle-weather.csv")


#filter data
data = data.dropna()


#features columns to train the model
features = ["precipitation", "temp_max", "temp_min", "wind"]

#Split the datasets into X and y
X = data[features]
y = data["weather"]

#Split the datasets into 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#train set - train model
#validation set - evaluate model
#test set - test model

#trainning the model
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42,
                                    class_weight = None,
                                    max_depth = 3,
                                    max_features = 'sqrt',
                                    max_leaf_nodes = None,
                                    min_samples_leaf = 1
                                    , min_samples_split = 2,
                                    min_weight_fraction_leaf = 0.0,
                                    splitter = 'best')

#fit trainnign data into model
dt_model.fit(X_train, y_train)


valueSend = {
    'model': dt_model
}

# # Save the model
joblib.dump(valueSend, 'decision_tree.pkl')
