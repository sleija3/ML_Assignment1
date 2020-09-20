import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

data = pd.read_csv("train_2.csv")
data_test = pd.read_csv("test_2.csv")

both_df = pd.concat([data, data_test], axis=0).reset_index(drop=True)

both_df.drop(["subject"], axis=1, inplace=True)
both_df = shuffle(both_df)


std_scaler = StandardScaler()
copy_both_df = both_df.copy()
X = both_df.loc[:, both_df.columns != "Activity"]
std_scaler.fit(X)
X_scaled = std_scaler.fit_transform(X)
y = both_df.Activity

y_encode = LabelEncoder().fit_transform(y)
labels = preprocessing.LabelEncoder().fit(y).classes_
#copy_both_df['Activity'] = LabelEncoder().fit_transform(copy_both_df.Activity)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encode, random_state=2)

# knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
num_estimators = [50, 100, 150, 200, 250, 300]
estimator_scores_train = []
estimator_scores_test = []
# ada = AdaBoostClassifier()
best_score = 0
best_score_estimator_val = 0
ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=4),
						n_estimators=250, learning_rate=.5)
for i in range(len(num_estimators)):
	n_estimator = num_estimators[i]
	ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=5),
						n_estimators=n_estimator, learning_rate=.5)
	ada.fit(X_train, y_train)
	y_predict_train = ada.predict(X_train)
	y_predict_test = ada.predict(X_test)
	estimator_scores_test.append(precision_score(y_test, y_predict_test, average='weighted'))
	estimator_scores_train.append(precision_score(y_train, y_predict_train, average='weighted'))
	if estimator_scores_test[i] > best_score:
		best_score = estimator_scores_test[i]
		best_score_estimator_val = num_estimators[i]

fig, ax = plt.subplots()
training = plt.plot(estimator_scores_train)
testing = plt.plot(estimator_scores_test)
plt.legend(['training', 'testing'])
plt.show()

ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=5),
						n_estimators=250, learning_rate=.5)
ada.fit(X_train, y_train)

# n_trees = len(ada)
# estimator_errors = ada.estimator_errors_[:n_trees]
#
#
# plt.figure(figsize=(15, 5))
#
# plt.subplot(131)
# plt.plot(range(1, n_trees + 1),
#          estimator_errors, c='black',
#          linestyle='dashed', label='SAMME.R')
# plt.legend()
# plt.ylim(0.18, 0.62)
# plt.ylabel('Test Error')
# plt.xlabel('Number of Trees')
# plt.show()

print("AdaBoost refitted with best score at N Iteration: ", best_score)
y_probas = ada.predict_proba(X_test)
y_temp = log_loss(y_test, y_probas)
print(y_temp)

print(" Score Train: ", ada.score(X_train, y_train))
print(" Score Test: ", ada.score(X_test, y_test))

metric_scores_test = ada.predict(X_test)
print(precision_score(y_test, metric_scores_test, average="weighted"))

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Testing Confusion matrix, without normalization", None),
                  ("Testing Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
	disp = plot_confusion_matrix(ada, X_test, y_test,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
	disp.ax_.set_title(title)

	print(title)
	print(disp.confusion_matrix)
	plt.show()

titles_options = [("Training Confusion matrix, without normalization", None),
                  ("Training Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
	disp = plot_confusion_matrix(ada, X_train, y_train,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
	disp.ax_.set_title(title)

	print(title)
	print(disp.confusion_matrix)
	plt.show()
