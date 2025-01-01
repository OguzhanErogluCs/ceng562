import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
import time
import warnings

warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')

# Read the training and test set
df1 = pd.read_csv("NSL-KDD-Dataset\\NSL_KDD_binary_train.csv")
df2 = pd.read_csv("NSL-KDD-Dataset\\NSL_KDD_binary_test.csv")

X_train = df1.iloc[:, :-1].values
y_train = df1.iloc[:, -1].values
X_test = df2.iloc[:, :-1].values
y_test = df2.iloc[:, -1].values

# print(df2)

# # KNN algorithm
# start_time = time.time()
# classifier = KNeighborsClassifier()
# classifier.fit(X_train, y_train)  # Model training
# y_pred = classifier.predict(X_test)  # Model testing
# print(classification_report(y_test, y_pred))
# print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
#
# # Plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# f, ax = plt.subplots(figsize=(5, 5))
# sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# end_time = time.time()
# print(f"Execution time: {end_time - start_time} seconds")
# plt.show()

# # Logistic regression
# start_time = time.time()
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(classification_report(y_test, y_pred))
# print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
#
# # Plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# f, ax = plt.subplots(figsize=(5, 5))
# sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# end_time = time.time()
# print(f"Execution time: {end_time - start_time} seconds")
# plt.show()

# # Random forest algorithm
# start_time = time.time()
# classifier = RandomForestClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(classification_report(y_test, y_pred))
# print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
#
# # Plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# f, ax = plt.subplots(figsize=(5, 5))
# sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# end_time = time.time()
# print(f"Execution time: {end_time - start_time} seconds")
# plt.show()

# # XGBoost algorithm
# start_time = time.time()
# classifier = XGBClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(classification_report(y_test, y_pred))
# print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
#
# # Plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# f, ax = plt.subplots(figsize=(5, 5))
# sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# end_time = time.time()
# print(f"Execution time: {end_time - start_time} seconds")
# plt.show()

# # LightGBM algorithm
# start_time = time.time()
# classifier = lgb.LGBMClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(classification_report(y_test, y_pred))
# print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
#
# # Plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# f, ax = plt.subplots(figsize=(5, 5))
# sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# end_time = time.time()
# print(f"Execution time: {end_time - start_time} seconds")
# plt.show()
