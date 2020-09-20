import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import plot_confusion_matrix, log_loss
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

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
labels = preprocessing.LabelEncoder().fit(y).classes_

y_encode = LabelEncoder().fit_transform(y)

copy_both_df['Activity'] = LabelEncoder().fit_transform(copy_both_df.Activity)

skf = StratifiedKFold(n_splits = 5)
dtc = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_split= 10, random_state = 2)

scoring = {'precision_weighted':make_scorer(precision_score, average='weighted'),
 		   'Accuracy': make_scorer(accuracy_score)}
parameters = {'max_depth': [5, 10, 15, 20],'min_samples_split':[10, 15, 17, 20, 25]}
clf = GridSearchCV(return_train_score=True, estimator=tree.DecisionTreeClassifier(),
				   param_grid=parameters, cv=skf, n_jobs=-1,
 				   scoring=scoring, refit='precision_weighted')

clf.fit(X=X_scaled, y=y_encode)
results = clf.cv_results_

plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("min_samples_split")
plt.ylabel("Score")

ax = plt.gca()
ax.set_xlim(2, 35)
ax.set_ylim(0, 1)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encode, random_state=2, test_size=.3, stratify=y_encode)
dtc.fit(X_train, y_train)

y_probas = dtc.predict_proba(X_test)
y_temp = log_loss(y_test, y_probas)
print("log loss: ", y_temp)

fig, ax = plt.subplots(figsize=(30, 30))
plot_tree(dtc, filled=True, fontsize=10)
plt.show()

print(" Score Train: ", dtc.score(X_train, y_train), " Tree Depth: ", dtc.get_depth())
print(" Score Test: ", dtc.score(X_test, y_test))

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Testing Confusion matrix, without normalization", None),
                  ("Testing Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
	disp = plot_confusion_matrix(dtc, X_test, y_test,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
	disp.ax_.set_title(title)
	plt.show()

	print(title)
	print(disp.confusion_matrix)

titles_options = [("Training Confusion matrix, without normalization", None),
                  ("Training Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
	disp = plot_confusion_matrix(dtc, X_train, y_train,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
	disp.ax_.set_title(title)
	plt.show()

	print(title)
	print(disp.confusion_matrix)


