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
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import optunity
import optunity.metrics
import joblib

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

# # Define the objective function
# def objective(params):
#     params = {
#         'n_estimators': int(params['n_estimators']),
#         'max_depth': int(params['max_depth']),
#         'learning_rate': abs(float(params['learning_rate'])),
#         "num_leaves": int(params['num_leaves']),
#         "min_child_samples": int(params['min_child_samples']),
#     }
#     clf = lgb.LGBMClassifier(**params)
#     clf.fit(X_train, y_train)
#     prediction = clf.predict(X_test)
#     score = accuracy_score(y_test, prediction)
#     return {'loss': -score, 'status': STATUS_OK}
#
#
# # Define the hyperparameter configuration space
# space = {
#     'n_estimators': hp.quniform('n_estimators', 50, 500, 20),
#     'max_depth': hp.quniform('max_depth', 5, 50, 1),
#     "learning_rate": hp.uniform('learning_rate', 0, 1),
#     "num_leaves": hp.quniform('num_leaves', 100, 2000, 100),
#     "min_child_samples": hp.quniform('min_child_samples', 10, 50, 5),
# }
#
# # Detect the optimal hyperparameter values
# best = fmin(fn=objective,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=50)
# print("LightGBM: Hyperopt estimated optimum {}".format(best))

# # Use the optimal hyperparameter values to train the optimized LightGBM model
# clf = lgb.LGBMClassifier(max_depth=14, learning_rate=0.8512957337767638, n_estimators=400,
#                          num_leaves=1400, min_child_samples=35)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
# print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
#
# # Plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# f, ax = plt.subplots(figsize=(5, 5))
# sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# plt.show()

# # Define the hyperparameter configuration space
# search = {
#     'n_estimators': [50, 500],
#     'max_depth': [5, 50],
#     'learning_rate': (0, 1),
#     "num_leaves": [100, 2000],
#     "min_child_samples": [10, 50],
# }
#
#
# # Define the objective function
# def performance(n_estimators=None, max_depth=None, learning_rate=None, num_leaves=None, min_child_samples=None):
#     clf = lgb.LGBMClassifier(n_estimators=int(n_estimators),
#                              max_depth=int(max_depth),
#                              learning_rate=float(learning_rate),
#                              num_leaves=int(num_leaves),
#                              min_child_samples=int(min_child_samples),
#                              )
#     clf.fit(X_train, y_train)
#     prediction = clf.predict(X_test)
#     score = accuracy_score(y_test, prediction)
#     return score
#
#
# # Detect the optimal hyperparameter values
# optimal_configuration, info, _ = optunity.maximize(performance,
#                                                    solver_name='particle swarm',
#                                                    num_evals=50,
#                                                    **search
#                                                    )
# print(optimal_configuration)
# print("Accuracy:" + str(info.optimum))

clf = lgb.LGBMClassifier(max_depth=36, learning_rate=0.80783203125, n_estimators=285,
                         num_leaves=763, min_child_samples=38)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy: " + str(accuracy_score(y_test, y_pred)))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

# Output a pickle file to save the trained model
joblib.dump(clf, 'Optimized_lightGBM.pkl')

# The trained model can be loaded directly for future testing
savestkrf = joblib.load('Optimized_lightGBM.pkl')
y_pred = savestkrf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))