
from os import path
project_folder = path.abspath(path.join(path.sep,"Projects", "PSiLON Net"))

# setup openml
import openml
with open(path.join(project_folder, "openml_apikey.txt")) as apikey_file:
    apikey = apikey_file.read()
openml.config.apikey = apikey
openml.config.cache_directory = path.join(project_folder, "datasets")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import itertools

def select_continuous_features(feature_names, categorical_indicator):
    features_categorical = itertools.compress(feature_names, categorical_indicator)
    return list(set(feature_names) - set(features_categorical))


n_keep = 10000
SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite
for task_id in benchmark_suite.tasks:  # iterate over all tasks
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    print("Loaded dataset: " + dataset.name)
    X, y, categorical_indicator, feature_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    X = X.iloc[:n_keep,:]
    y = y.iloc[:n_keep]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    features_continuous = select_continuous_features(feature_names, categorical_indicator)
    transformer = ColumnTransformer([('scaler', StandardScaler(), features_continuous)],
                                    remainder = 'passthrough')
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    if SUITE_ID == 335:
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train = (y_train-y_mean)/y_std
        y_test = (y_test-y_mean)/y_std


######################################################################################

from sklearn.ensemble import RandomForestRegressor as RFR
RMSE = lambda Y,Y_hat: np.sqrt(np.mean((Y-Y_hat)**2))

model = RFR(n_estimators = 2000, 
            min_samples_leaf = 8)
model.fit(X_train, Y_train)
Y_hat_val = model.predict(X_val)
RMSE(Y_val, Y_hat_val)

Y_hat_test = model.predict(X_test)
RMSE(Y_test, Y_hat_test)
# 0.411413

from sklearn.linear_model import Ridge

model = Ridge(alpha = 5.1)
model.fit(X_train, Y_train)
Y_hat_val = model.predict(X_val)
RMSE(Y_val, Y_hat_val)

Y_hat_test = model.predict(X_test)
RMSE(Y_test, Y_hat_test)
# 0.465901


#######################################################################################
import torch

X_train = torch.from_numpy(X_train).to(torch.float32)
X_val = torch.from_numpy(X_val).to(torch.float32)
X_test = torch.from_numpy(X_test).to(torch.float32)
Y_train = torch.from_numpy(Y_train).to(torch.float32)
Y_val = torch.from_numpy(Y_val).to(torch.float32)
Y_test = torch.from_numpy(Y_test).to(torch.float32)

