import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix, log_loss
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


#data = pd.read_csv("train.csv")
data = pd.read_csv("bbc-text.csv")
data['text_clean'] = data['text'].apply(nltk.word_tokenize)
print('done tokenize')
stop_words=set(nltk.corpus.stopwords.words("english"))
data['text_clean'] = data['text_clean'].apply(lambda x: [item for item in x if item not in stop_words])
regex = '[a-z]+'
print('done stop words')
data['text_clean'] = data['text_clean'].apply(lambda x: [item for item in x if re.match(regex, item)])
lem = nltk.stem.wordnet.WordNetLemmatizer()
data['text_clean'] = data['text_clean'].apply(lambda x: [lem.lemmatize(item, pos='v') for item in x])
print("done lemmatizer")


X = data.loc[:, data.columns != "category"]
enc = LabelEncoder()
y = enc.fit_transform(data['category'])
labels = list(enc.classes_)
print(data.category.value_counts())

X_train, X_test, y_train, y_test = train_test_split(data['text'], y, test_size=.2, stratify=y)
vec = TfidfVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

print('Vectorization complete.\n')
print()
# dtc.fit(X_train, y_train)

nn = MLPClassifier(activation='logistic', early_stopping=True, shuffle=True,
				   learning_rate='adaptive', hidden_layer_sizes=(1200), n_iter_no_change=30, learning_rate_init=.3)
# scores = cross_val_score(dtc, X_train_vec, y_train, cv=5, scoring="precision_weighted")

# parameters = {'activation':['logistic'], 'early_stopping':[True], 'shuffle':[True],
# 			  'hidden_layer_sizes':[10, 25, 50, 100, 150], 'learning_rate':['adaptive']}
# parameters = {'hidden_layer_sizes': [10, 25, 50, 100, 125, 150, 200], 'activation':['logistic'],
# 			  'learning_rate':['adaptive'], 'solver':['sgd'], 'learning_rate_init':[1]}
# scoring = {'f1_weighted':make_scorer(f1_score, average='weighted')}
# skf = StratifiedKFold(n_splits = 5)
# clf = GridSearchCV(MLPClassifier(), parameters, cv=skf, n_jobs=-1)
# clf.fit(X=X_train_vec, y=y_train)
# results = clf.cv_results_

# print(results['mean_test_f1_weighted'])
# print (clf.best_score_, clf.best_params_)
# num_parameters = [10, 25, 50, 100, 125, 150, 200, 250, 300, 400]

num_parameters = [900, 1200, 1500, 1800]
estimator_scores_train = []
estimator_scores_test = []
best_score = 0
best_score_estimator_val = 0
for i in range(len(num_parameters)):
	parameter = num_parameters[i]
	nn = MLPClassifier(activation='logistic', early_stopping=True, shuffle=True,
 				   learning_rate='adaptive', hidden_layer_sizes=parameter)
	nn.fit(X_train_vec, y_train)
	y_predict_train = nn.predict(X_train_vec)
	y_predict_test = nn.predict(X_test_vec)
	estimator_scores_test.append(f1_score(y_test, y_predict_test, average='weighted'))
	estimator_scores_train.append(f1_score(y_train, y_predict_train, average='weighted'))
	if estimator_scores_test[i] > best_score:
		best_score = estimator_scores_test[i]
		best_score_estimator_val = num_parameters[i]
	print("hi")

fig, ax = plt.subplots()
training = plt.plot(estimator_scores_train)
testing = plt.plot(estimator_scores_test)
plt.legend(['training', 'testing'])
plt.show()


#
# print(estimator_scores_train)
# print(estimator_scores_test)


#
# plt.figure(figsize=(13, 13))
# plt.title("GridSearchCV evaluating Neural Network",
#           fontsize=16)
#
# plt.xlabel("Layer Size")
# plt.ylabel("Score")
#
# ax = plt.gca()
# ax.set_xlim(10, 200)
# ax.set_ylim(0, 1)
#
# # Get the regular numpy array from the MaskedArray
# X_axis = np.array(results['param_hidden_layer_sizes'].data, dtype=float)
#
# for scorer, color in zip(sorted(scoring), ['g', 'k']):
# 	for sample, style in (('train', '--'), ('test', '-')):
# 		sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
# 		sample_score_std = results['std_%s_%s' % (sample, scorer)]
# 		ax.fill_between(X_axis, sample_score_mean - sample_score_std,
#                         sample_score_mean + sample_score_std,
#                         alpha=0.1 if sample == 'test' else 0, color=color)
# 		ax.plot(X_axis, sample_score_mean, style, color=color,
#                 alpha=1 if sample == 'test' else 0.7,
#                 label="%s (%s)" % (scorer, sample))
#
# 	best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
# 	best_score = results['mean_test_%s' % scorer][best_index]
#
# 	# Plot a dotted vertical line at the best score for that scorer marked by x
# 	ax.plot([X_axis[best_index], ] * 2, [0, best_score],
#             linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
#
# 	# Annotate the best score for that scorer
# 	ax.annotate("%0.2f" % best_score,
#                 (X_axis[best_index], best_score + 0.005))
#
# plt.legend(loc="best")
# plt.grid(False)
# plt.show()

nn.fit(X_train_vec, y_train)

fig1, ax2 = plt.subplots()
ax2.plot(nn.loss_curve_)
plt.show()

y_probas = nn.predict_proba(X_test_vec)
y_temp = log_loss(y_test, y_probas)
print("Log Loss: ", y_temp)

print(" Score Train: ", nn.score(X_train_vec, y_train))
print(" Score Test: ", nn.score(X_test_vec, y_test))
metric_scores_test = nn.predict(X_test_vec)
print(f1_score(y_test, metric_scores_test, average="weighted"))
#
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Testing Confusion matrix, without normalization", None),
                  ("Testing Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
	disp = plot_confusion_matrix(nn, X_test_vec, y_test,
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
	disp = plot_confusion_matrix(nn, X_train_vec, y_train,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
	disp.ax_.set_title(title)

	print(title)
	print(disp.confusion_matrix)
	plt.show()



