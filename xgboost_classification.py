# XGBoost blue print for classification with cross validation and parameter search
# Read In Data
import pandas as pd
data = pd.read_csv('139394485_T_T100D_MARKET_ALL_CARRIER.csv')

# Define col names for the parameters of the network
pred_vars = ['MONTH', 'ORIGIN', 'DEST','DISTANCE']
target_var = 'PASSENGERS'
keep = pred_vars
keep.append(target_var)

# Subset only what's needed
data = data[keep]

def get_labels(data_list, cut_off = None):
    if cut_off is None:
        from statistics import mean
        cut_off = mean(data_list)
        print('using cut off : ', cut_off)
    output = [1 if dl >= cut_off else 0 for dl in data_list]
    return output


# Encode the  source and target nodes using a catagory encoder
from category_encoders import OneHotEncoder
ce = OneHotEncoder()
ce.fit(data)

# transform the encoded data
data_encoded = ce.transform(data)
labels = get_labels(data[target_var])
data_encoded.drop(target_var, 1, inplace=True)

# split out a final eval set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_encoded, labels, random_state=0, test_size=.25)

# convert to xgb data format
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# create a baseline model using GaussianNB Classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
preds = gnb.predict(X_test)

# calculate AUC on the test set using the baseline classifier
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds, pos_label=1)
metrics.auc(fpr, tpr)

# set up params for xgboost to
params = {'max_depth':6,
          'min_child_weight': 1,
          'eta': .3,
          'subsample': 1,
          'colsample_bytree': 1,
          'scale_pos_weight':1,
          'objective':'binary:logistic',
          'eval_metric':"auc"}

num_boost_round = 20

# build the model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10)

# build a cross validated model with no parameter search
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'auc'},
    early_stopping_rounds=10
)

# Use a hyper parameter search using brute force method with cross validation
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(3,5)
    for min_child_weight in range(5,8)]

max_auc = float(0)
best_params = None

for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))

    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'auc'},
        early_stopping_rounds=10
    )

    # Update best AUC
    print(cv_results)
    mean_auc = cv_results['test-auc-mean'].max()
    boost_rounds = cv_results['test-auc-mean'].argmax()
    print("\tAUC {} for {} rounds".format(max_auc, boost_rounds))
    if mean_auc > max_auc:
        max_auc = mean_auc
        best_params = (max_depth, min_child_weight)

print("Best params: max_depth {}, min_child_weight {}, Max AUC: {}".format(best_params[0], best_params[1], max_auc))

# set the params found as best in the last step
params['max_depth'] = 4
params['min_child_weight'] = 5

# re train the model with the new found parameters
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round+200,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)
print("Best AUC: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))

num_boost_round = model.best_iteration + 1

# re train the model to only the optimum iteration (it usually has been over trained
best_model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")]
)
# final check
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, best_model.predict(dtest), pos_label=1)
print(metrics.auc(fpr, tpr))

# save the best model
model_file_name = 'best_xgb_class.model'
best_model.save_model(model_file_name)

# save the encoder
import pickle
import xgboost as xgb
import numpy as np
import pandas as bd

encoder_file_name = 'train_cat_encoder.p'
pickle.dump(ce, open(encoder_file_name, 'wb'))

var_file_name = 'varNames.p'
pickle.dump(keep, open(var_file_name, 'wb'))

# load the saved model
loaded_model = xgb.Booster()
loaded_model.load_model(model_file_name)

# load the pre trained cat encoder
fileObject = open(encoder_file_name,'rb')
loaded_ce = pickle.load(fileObject)

# load the variable names
fileObject = open(var_file_name, 'rb')
loaded_varNames = pickle.load(fileObject)

# Use the encoder and the pred model for predictions.
new_data =loaded_ce.transform(data[loaded_varNames])
preds = loaded_model.predict(xgb.DMatrix(new_data))


# Visualization Section

# plot a single Tree
from xgboost import plot_tree
plot_tree(best_model, num_trees=0, rankdir='LR')

#plot variable Importance
import seaborn as sns
sns.set(font_scale = 1.5)
from xgboost import plot_importance
importances =best_model.get_fscore()
importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace=True)
importance_frame = importance_frame.tail(20)
importance_frame.plot(kind = 'barh', x = 'Feature', figsize=(8,8), color='blue')


# plot AUC Curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
preds = best_model.predict(dtest)
labels = y_test
fpr, tpr, _ = roc_curve(labels, preds)
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print('ROC AUC: %0.2f' % roc_auc)

# Plot of a ROC curve for a specific class
import scikitplot as skplt
import matplotlib.pyplot as plt

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Sense Spec Curve
import pylab as pl
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


# get a regression report
from sklearn.metrics import classification_report
target_names = ['less and average', 'over average']
print(classification_report(labels, preds, target_names=target_names))