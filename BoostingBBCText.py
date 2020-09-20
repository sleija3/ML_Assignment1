import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix, log_loss
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer

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

# data = shuffle(data)
# std_scaler = StandardScaler()
# copy_both_df = data.copy()
X = data.loc[:, data.columns != "category"]
# std_scaler.fit(X)
# X_scaled = std_scaler.fit_transform(X)
enc = LabelEncoder()
y = enc.fit_transform(data['category'])
labels = list(enc.classes_)

X_train, X_test, y_train, y_test = train_test_split(data['text'], y, test_size=.2, stratify=y)
vec = TfidfVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

print('Vectorization complete.\n')
ada = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(criterion='entropy', max_depth=5),
                         n_estimators=750, learning_rate=.5)

# num_estimators = [550, 650, 750, 850, 950, 1050, 1150]
num_estimators = [550, 650, 750, 850]
estimator_scores_train = []
estimator_scores_test = []
best_score = 0
best_score_estimator_val = 0
for i in range(len(num_estimators)):
	n_estimator = num_estimators[i]
	ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=5),
						n_estimators=n_estimator, learning_rate=.5)
	ada.fit(X_train_vec, y_train)
	y_predict_train = ada.predict(X_train_vec)
	y_predict_test = ada.predict(X_test_vec)
	estimator_scores_test.append(f1_score(y_test, y_predict_test, average='weighted'))
	estimator_scores_train.append(f1_score(y_train, y_predict_train, average='weighted'))
	if estimator_scores_test[i] > best_score:
		best_score = estimator_scores_test[i]
		best_score_estimator_val = num_estimators[i]

fig, ax = plt.subplots()
training = plt.plot(estimator_scores_train)
testing = plt.plot(estimator_scores_test)
plt.legend(['training', 'testing'])
plt.show()

print(estimator_scores_train)
print(estimator_scores_test)

ada.fit(X_train_vec, y_train)
y_probas = ada.predict_proba(X_test_vec)
y_temp = log_loss(y_test, y_probas)
print(y_temp)

print(" Score Train: ", ada.score(X_train_vec, y_train))
print(" Score Test: ", ada.score(X_test_vec, y_test))

metric_scores_test = ada.predict(X_test_vec)
print(f1_score(y_test, metric_scores_test, average="weighted"))

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
	disp = plot_confusion_matrix(ada, X_test_vec, y_test,
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
	disp = plot_confusion_matrix(ada, X_train_vec, y_train,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
	disp.ax_.set_title(title)

	print(title)
	print(disp.confusion_matrix)
	plt.show()

print(ada.feature_importances_[0:400])
