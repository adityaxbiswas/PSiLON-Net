
from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# creating column transformer (this will help us normalize/preprocess our data)
data_folder = path.abspath(path.join(path.sep,"Projects", "PathNet"))
df = pd.read_csv(path.join(data_folder, "insurance.csv"))
ct = make_column_transformer((StandardScaler(), ['age', 'bmi', 'children']), 
                             (OneHotEncoder(handle_unknown = 'ignore'), ['region']),
                             remainder = 'passthrough')
X = df.drop('charges', axis = 1)
X['smoker'] = (X['smoker'] == 'yes')*1
X['sex'] = (X['sex'] == 'male')*1
Y = np.log(df['charges'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, shuffle = False)
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.25,
                                                  shuffle = False)

########################################################################################


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

