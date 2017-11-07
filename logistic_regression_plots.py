import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
# load data
data_all, labels_all = load_breast_cancer(True)

# split data
X_train, X_test, y_train, y_test = train_test_split(data_all, labels_all, random_state=0, test_size=.10)

# make AUC the scorer
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                  param_grid={'min_samples_split': range(2, 403, 10)},
                  scoring=scoring, cv=5, refit='AUC')
gs.fit(X_train, y_train)
results = gs.cv_results_

########################################
# Plot CV Parameter Search for
#   Scorer
########################################

plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("min_samples_split")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, 402)
ax.set_ylim(0.73, 1)

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
plt.grid('off')
plt.show()



######################################
# plot AUC Curve
#######################################

from sklearn.metrics import roc_curve, auc
# get tpr/fpr for test set
preds = gs.predict(X_test)
labels = y_test
fpr, tpr, _ = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)
print('Test ROC AUC: %0.2f' % roc_auc)


# get tpr/fpr for train set
preds_train = gs.predict(X_train)
labels_train = y_train
fpr_train, tpr_train, _ = roc_curve(labels_train, preds_train)
roc_auc_train = auc(fpr_train, tpr_train)
print(' Train ROC AUC: %0.2f' % roc_auc_train)


# Plot of a ROC curve for a specific class
import matplotlib.pyplot as plt

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='Test ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr_train, tpr_train, label='Train ROC curve (area = %0.2f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


##########################
# Plot Sense Spec Curve
##########################

import pylab as pl
import pandas as pd
fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.ix[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'])
pl.plot(roc['1-fpr'], color = 'red')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])


##########################################################
# Plot a Decision Boundary against principal components
###########################################################

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train)
prince_comp_train = pca.fit_transform(X_train)

# generate grid for plotting
h = 10
x_min, x_max = prince_comp_train[:,0].min() - 1, prince_comp_train[:, 0].max() + 1
y_min, y_max = prince_comp_train[:,1].min() - 1, prince_comp_train[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]

new_grid = pca.inverse_transform(grid)
# create decision boundary plot
Z = gs.predict(new_grid)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=matplotlib.cm.get_cmap('Spectral'), alpha=0.5)
plt.scatter(prince_comp_train[:,0], prince_comp_train[:, 1], c=y_train, marker='.', cmap=matplotlib.cm.get_cmap('Spectral'))
plt.title('Decision Boundary by Principal Components')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.show()



