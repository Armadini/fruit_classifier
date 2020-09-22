
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


#CREATE DATA FRAME
fruits = pd.read_table("fruit_data.txt")
print(fruits.head())

print(fruits.shape)
print(fruits['fruit_name'].unique())
print(fruits.groupby('fruit_name').size())

# import seaborn as sns
# sns.countplot(fruits['fruit_name'], label="Count")
# plt.show()

fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), \
            title='Box Plot for each input var')

plt.savefig('fruits_box')
# plt.show()

import pylab as pl
fruits.drop('fruit_label', axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Histogram for each numeric int var")
plt.savefig("fruits_hist")
# plt.show()

feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']

#SPLIT DATASET
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print(f"LOGREG accuracy on TRAINING data: {logreg.score(X_train, y_train)}")
print(f"LOGREG accuracy on TESTING data: {logreg.score(X_test, y_test)}")


#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

print(f"DECISION TREE accuracy on TRAINING data: {clf.score(X_train, y_train)}")
print(f"DECISION TREE accuracy on TESTING data: {clf.score(X_test, y_test)}")


#KNN CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
base_knn = KNeighborsClassifier()
parameters = {'n_neighbors': [1 + 2**x for x in range(5)], 'weights': ['distance']}

knn = GridSearchCV(base_knn, parameters, cv=3)
knn.fit(X_train, y_train)
print('Best Hyperparameters: ', knn.best_params_, '\n')

print(f"KNN accuracy on TRAINING data: {knn.score(X_train, y_train)}")
print(f"KNN accuracy on TESTING data: {knn.score(X_test, y_test)}")


#GAUSSIAN NAIVE BAYES CLASSIFIER
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

print(f"NAIVE BAYES accuracy on TRAINING data: {gnb.score(X_train, y_train)}")
print(f"NAIVE BAYES accuracy on TESTING data: {gnb.score(X_test, y_test)}")


#SUPPORT VECTOR MACHINE CLASSIFIER
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)

print(f"SVM accuracy on TRAINING data: {svm.score(X_train, y_train)}")
print(f"SVM accuracy on TESTING data: {svm.score(X_test, y_test)}")

gamma = 'auto'
#CONFUSION MATRICIES
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

knn_pred = knn.predict(X_test)
print(f"KNN CONFUSION MATRIX:\n{confusion_matrix(y_test, knn_pred)}")
print(classification_report(y_test, knn_pred))

