# XGBoost blue print for regression with cross validation and parameter search
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

# Encode the  source and target nodes using a catagory encoder
from category_encoders import OneHotEncoder
ce = OneHotEncoder()
ce.fit(data)

# transform the encoded data
data_encoded = ce.transform(data)
labels = data[target_var]
data_encoded.drop(target_var, 1, inplace=True)

# split out a final eval set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_encoded, labels, random_state=0, test_size=.25)

# convert to xgb data format
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# create a baseline model
from sklearn.metrics import mean_absolute_error
import numpy as np

# "Learn" the mean from the training data
mean_train = np.mean(y_train)

# Get predictions on the test set
baseline_predictions = np.ones(y_test.shape) * mean_train

# Compute MAE
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))

# set up params for xgboost to
params = {'max_depth':6,
          'min_child_weight': 1,
          'eta': .3,
          'subsample': 1,
          'colsample_bytree': 1,
          'objective':'reg:linear',
          'eval_metric':"mae"}
num_boost_round = 999

# build the model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10)

# build a cross validated model
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
)



# Use a hyper parameter search using brute force method
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(3,5)
    for min_child_weight in range(7,8)]

min_mae = float("Inf")
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
        metrics={'mae'},
        early_stopping_rounds=10
    )

    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

# set the params found as best in the last step
params['max_depth'] = 9
params['min_child_weight'] = 9

# re train the model with the new found parameters
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)
print("Best MAE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))

num_boost_round = model.best_iteration + 1

# re train the model to only the optimum iteration (it usually has been over trained
best_model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")]
)
# final check
mean_absolute_error(best_model.predict(dtest), y_test)

# save the best model
best_model.save_model("my_model.model")

# load the saved model
loaded_model = xgb.Booster()
loaded_model.load_model("my_model.model")

# And use it for predictions.
loaded_model.predict(dtest)


# Visualization Section
# plot a single Tree
from xgboost import plot_tree
plot_tree(best_model, num_trees=0, rankdir='LR')

# plot var imp
%matplotlib inline
import seaborn as sns
sns.set(font_scale = 1.5)


from xgboost import plot_importance
importances =best_model.get_fscore()
importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)
importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'blue')


# regression scatter plot
import seaborn as sns
import scipy
labs = ['actual', 'predicted']
results_df = pd.DataFrame({labs[0]: y_test,  labs[1]: best_model.predict(dtest)})
xlims = np.min(results_df[labs[0]]), np.max(results_df[labs[0]])/2
ylims = xlims

sns.set(style="darkgrid", color_codes=True)
g = sns.jointplot(labs[0], labs[1], data=results_df, kind="scatter" ,
                  xlim=xlims, ylim=ylims, color="b", size=10)
