import matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from river import metrics
from river import stream
import matplotlib.pyplot as plt
import seaborn as sns
import time
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import optunity
import optunity.metrics
import warnings


matplotlib.use("TkAgg")

warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv("NSL-KDD-Dataset\\NSL_KDD_binary(train+test).csv")
df1 = pd.read_csv("NSL-KDD-Dataset\\NSL_KDD_binary_train.csv")
df2 = pd.read_csv("NSL-KDD-Dataset\\NSL_KDD_binary_test.csv")

df_used = int(len(df1) * 0.1) + len(df2)
df0 = df.iloc[-df_used:]

# print(df0)

X = df0.drop(['label'], axis=1)
y = df0['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=0.9, shuffle=False, random_state=0)

classifier = lgb.LGBMClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)


# print("Accuracy: "+str(accuracy_score(y_test,predictions)))


# def oasw(a=0.99, b=0.95, win1=200, win2=1000):
#     a = round(a, 3)
#     b = round(b, 3)
#     win1 = round(win1)
#     win2 = round(win2)
#
#     i = 0
#     yt = []
#     yp = []
#     x_new = []
#     y_new = []
#     dr = [0]
#     d = 0
#     f = 0
#     tt = 0
#     th = 0
#     xt = []
#
#     metric = metrics.Accuracy()
#
#     classifier = lgb.LGBMClassifier()  # Train the initial LightGBM model
#     classifier.fit(X_train, y_train)
#
#     for xi, yi in stream.iter_pandas(X_test, y_test):
#
#         xi2 = np.array(list(xi.values()))
#         y_pred = classifier.predict(xi2.reshape(1, -1))[0]  # make a prediction
#         metric.update(yi, y_pred)  # update the metric
#
#         # Store the y_test, y_pred, and x_test
#         yt.append(yi)
#         yp.append(y_pred)
#         xt.append(xi2)
#
#         # Monitor the accuracy changes in the sliding window
#         if i > 2 * win1:
#             acc1 = accuracy_score(yt[i - win1:], yp[i - win1:])  # Current window accuracy
#             acc2 = accuracy_score(yt[i - 2 * win1:i - win1], yp[i - 2 * win1:i - win1])  # Last window accuracy
#             if (d == 0) & (acc1 < a * acc2):  # If the window accuracy drops to the warning level
#                 x_new.append(xi2)
#                 y_new.append(yi)
#                 d = 1
#             if d == 1:  # In the warning level
#                 tt = len(y_new)
#                 if acc1 < b * acc2:  # If the window accuracy drops to the drift level
#                     dr.append(i)  # Record the drift start point
#                     f = i
#                     if tt < win1:  # if enough new concept samples are collected
#                         classifier.fit(xt[i - win1:], yt[i - win1:])
#                     else:
#                         classifier.fit(x_new, y_new)
#                     d = 2
#                 elif (acc1 > a * acc2) | (
#                         tt == win2):  # If the window accuracy increases back to the normal level (false alarm)
#                     x_new = []
#                     y_new = []
#                     d = 0
#                 else:
#                     x_new.append(xi2)
#                     y_new.append(yi)
#             if d == 2:  # In the drift level
#                 tt = len(y_new)
#                 acc3 = accuracy_score(yt[f:f + win1], yp[f:f + win1])
#                 x_new.append(xi2)
#                 y_new.append(yi)
#                 if tt >= win1:
#                     if (acc1 < a * acc3):  # When new concept accuracy drops to the warning level
#                         if th == 0:
#                             classifier.fit(x_new,
#                                            y_new)  # Retrain the classifier on all the newly collected samples to obtain a robust classifier
#                             th = 1
#                     if (th == 1) & (tt == win2):  # When sufficient new concept samples are collected
#                         classifier.fit(x_new, y_new)  # obtain a robust classifier
#                         x_new = []
#                         y_new = []
#                         d = 0  # Go back to the normal state for next potential drift detection
#                         th = 0
#
#         i = i + 1
#     score = metric.get()
#     print(str(a) + " " + str(b) + " " + str(win1) + " " + str(win2) + " " + str(
#         score))  # Output the hyperparameter values and corresponding accuracy
#     return score
#
#
# search = {
#     'a': [0.95, 0.99],
#     'b': [0.90, 0.98],
#     'win1': [200, 1000],
#     'win2': [1000, 5000],
# }
#
# optimal_configuration, info, _ = optunity.maximize(oasw,
#                                                    solver_name='particle swarm',
#                                                    num_evals=10,
#                                                    **search
#                                                    )
# print(optimal_configuration)
# print("Accuracy:" + str(info.optimum))


# Define OASW with a figure
def oasw_plot(a=0.99, b=0.95, win1=200, win2=1000):
    a = round(a, 3)
    b = round(b, 3)
    win1 = round(win1)
    win2 = round(win2)

    metric = metrics.Accuracy()
    metric2 = metrics.Accuracy()

    i = 0
    t = []
    yt = []
    yp = []
    m = []
    m2 = []
    x_new = []
    y_new = []
    dr = [0]
    d = 0
    f = 0
    tt = 0
    th = 0
    xt = []

    classifier = lgb.LGBMClassifier()  # Train the initial LightGBM model
    classifier.fit(X_train, y_train)

    classifier2 = lgb.LGBMClassifier()  # Train an offline LightGBM model as a comparison model
    classifier2.fit(X_train, y_train)

    for xi, yi in stream.iter_pandas(X_test, y_test):

        xi2 = np.array(list(xi.values()))
        y_pred = classifier.predict(xi2.reshape(1, -1))[0]  # make a prediction
        metric.update(yi, y_pred)  # update the metric

        y_pred2 = classifier2.predict(xi2.reshape(1, -1))[0]
        metric2.update(yi, y_pred2)

        # Store the y_test, y_pred, x_test, and real-time accuracy
        t.append(i)
        m.append(metric.get() * 100)
        yt.append(yi)
        yp.append(y_pred)
        m2.append(metric2.get() * 100)
        xt.append(xi2)

        # Monitor the accuracy changes in the sliding window
        if i > 2 * win1:
            acc1 = accuracy_score(yt[i - win1:], yp[i - win1:])  # Current window accuracy
            acc2 = accuracy_score(yt[i - 2 * win1:i - win1], yp[i - 2 * win1:i - win1])  # Last window accuracy
            if (d == 0) & (acc1 < a * acc2):  # If the window accuracy drops to the warning level
                x_new.append(xi2)
                y_new.append(yi)
                d = 1
            if d == 1:  # In the warning level
                tt = len(y_new)
                if acc1 < b * acc2:  # If the window accuracy drops to the drift level
                    dr.append(i)  # Record the drift start point
                    f = i
                    if tt < win1:  # if enough new concept samples are collected
                        classifier.fit(xt[i - win1:], yt[i - win1:])
                    else:
                        classifier.fit(x_new, y_new)
                    d = 2
                elif (acc1 > a * acc2) | (
                        tt == win2):  # If the window accuracy increases back to the normal level (false alarm)
                    x_new = []
                    y_new = []
                    d = 0
                else:
                    x_new.append(xi2)
                    y_new.append(yi)

            if d == 2:  # In the drift level
                tt = len(y_new)
                acc3 = accuracy_score(yt[f:f + win1], yp[f:f + win1])
                x_new.append(xi2)
                y_new.append(yi)
                if tt >= win1:
                    if (acc1 < a * acc3):  # When new concept accuracy drops to the warning level
                        if th == 0:
                            classifier.fit(x_new,
                                           y_new)  # Retrain the classifier on all the newly collected samples to obtain a robust classifier
                            th = 1
                    if (th == 1) & (tt == win2):  # When sufficient new concept samples are collected
                        classifier.fit(x_new, y_new)  # obtain a robust classifier
                        x_new = []
                        y_new = []
                        d = 0  # Go back to the normal state for next potential drift detection
                        th = 0

        i = i + 1

    # Plot the accuracy change figure
    plt.rcParams.update({'font.size': 35})
    plt.ion()
    plt.figure(1, figsize=(24, 15))
    sns.set_style("darkgrid")
    plt.clf()
    plt.plot(t, m, '-b', label='OASW+LightGBM, Avg Accuracy: %.2f%%' % (metric.get() * 100))
    plt.plot(t, m2, 'red', label='Offline LightGBM, Avg Avg Accuracy: %.2f%%' % (metric2.get() * 100))

    # Plot the drift points
    for i in range(len(dr)):
        if i != 0:
            plt.scatter(dr[i], m[dr[i]], s=200, c='r')

    plt.legend(loc='best')
    plt.ylim(80, 101)
    plt.title('NSL-KDD', fontsize=40)
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy(%)')

    plt.show(block=True)


# Assign the optimal hyperparameters detected by PSO
oasw_plot(a=0.981, b=0.947, win1=605, win2=3779)
