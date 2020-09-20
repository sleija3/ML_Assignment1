import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import make_scorer, plot_confusion_matrix, log_loss
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

# data = shuffle(data)
# std_scaler = StandardScaler()
# copy_both_df = data.copy()
X = data.loc[:, data.columns != "category"]
# std_scaler.fit(X)
# X_scaled = std_scaler.fit_transform(X)
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

knn = KNeighborsClassifier(weights='distance', n_neighbors=8)

parameters = {'n_neighbors':[1, 2, 4, 8, 16, 20, 40], 'weights':['distance']}
# parameters = {'max_depth':range(3, 20)}
scoring = {'f1_weighted':make_scorer(f1_score, average='weighted')}
skf = StratifiedKFold(n_splits = 5)
clf = GridSearchCV(return_train_score=True, estimator=KNeighborsClassifier(),
 				   param_grid=parameters, cv=skf, n_jobs=-1, scoring=scoring, refit='f1_weighted')
clf.fit(X=X_train_vec, y=y_train)
results = clf.cv_results_
print (clf.best_score_, clf.best_params_)
print(results['mean_test_f1_weighted'])


plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating scores (increments of 5 neighbors)",
          fontsize=16)

plt.xlabel("n_neighbors")
plt.ylabel("Score")

ax = plt.gca()
ax.set_xlim(1, 40)
ax.set_ylim(.85, 1)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_n_neighbors'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
	for sample, style in (('train', '--'), ('test', '-')):
		sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
		sample_score_std = results['std_%s_%s' % (sample, scorer)]
		ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
		ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

	best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
	best_score = results['mean_test_%s' % scorer][best_index]

	# Plot a dotted vertical line at the best score for that scorer marked by x
	ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

	# Annotate the best score for that scorer
	ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()

knn.fit(X_train_vec, y_train)

y_probas = knn.predict_proba(X_test_vec)
y_temp = log_loss(y_test, y_probas)
print("Log Loss: ", y_temp)

print(" Score Train: ", knn.score(X_train_vec, y_train))
print(" Score Test: ", knn.score(X_test_vec, y_test))
metric_scores_test = knn.predict(X_test_vec)
print(f1_score(y_test, metric_scores_test, average="weighted"))
#
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Testing Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
	disp = plot_confusion_matrix(knn, X_test_vec, y_test,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
	disp.ax_.set_title(title)
	plt.show()
	print(title)
	print(disp.confusion_matrix)

titles_options = [("Training Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
	disp = plot_confusion_matrix(knn, X_train_vec, y_train,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
	disp.ax_.set_title(title)
	plt.show()
	print(title)
	print(disp.confusion_matrix)



