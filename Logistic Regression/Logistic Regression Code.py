import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Getting Data
ad_data = pd.read_csv("advertising.csv")
print(ad_data.head())
print()
print(ad_data.describe())
print()
print(ad_data.info())
print()

# EDA
ad_data["Age"].hist(bins = 30)
plt.xlabel("Age")

sns.jointplot(x = "Age", y = "Area Income", data = ad_data)

sns.jointplot(x = "Age", y = "Daily Time Spent on Site", data = ad_data, kind = "kde",
              space = 0, fill = True, thresh = 0, cmap = 'Blues')

sns.jointplot(x = "Daily Time Spent on Site", y = "Daily Internet Usage", data = ad_data)

sns.pairplot(ad_data, hue = "Clicked on Ad", palette = "magma")

# Training Model
print(ad_data.columns)
print()

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage','Male']] # Choosing only Numerical Columns
y = ad_data["Clicked on Ad"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Predictions and Evaluation
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print()
print(classification_report(y_test, predictions))




























