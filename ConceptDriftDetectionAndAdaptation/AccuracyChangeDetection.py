import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
import matplotlib

warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')

df = pd.read_csv("NSL-KDD-Dataset\\NSL_KDD_binary(train+test).csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=0.9, random_state=0, shuffle=False)

classifier = lgb.LGBMClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy: " + str(accuracy_score(y_test, y_pred)))

# # Plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# f, ax = plt.subplots(figsize=(5, 5))
# sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# plt.show()

# Record the real-time accuracy based on timestamp/sample index
acc = []
acc_sum = 0
for i in range(0, len(y_test)):
    if y_test[i] == y_pred[i]:
        acc_sum = acc_sum + 1
    accuracy = acc_sum/(i+1)
    acc.append(accuracy)

# Plot the accuracy changes
plt.rcParams.update({'font.size': 20})
plt.figure(1,figsize=(24,15))
plt.clf()
plt.plot(acc,'-b',label='LightGBM Accuracy')

df1 = pd.read_csv("NSL-KDD-Dataset\\NSL_KDD_binary_train.csv")
df2 = pd.read_csv("NSL-KDD-Dataset\\NSL_KDD_binary_test.csv")

plt.scatter(len(df1)-len(X_train),acc[len(X_test)-len(df2)],s=100,c='r')
plt.text(len(df1)-len(X_train),acc[len(X_test)-len(df2)]+0.0005, 'test set starts', c='r')

plt.legend(loc='best')
plt.title('NSL-KDD Intrusion Detection Accuracy Change', fontsize=20)
plt.xlabel('Timestamp')
plt.ylabel('Accuracy')

plt.show()