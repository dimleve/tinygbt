import pandas as pd
from sklearn.metrics import mean_squared_error

from tinygbt import Dataset, GBT

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

def inverse_logit_function(input):
    return 1.0 / (1.0 + np.exp(-input))

print('Load data...')

data = pd.read_csv('./data/data_banknote_authentication.txt', header=None)

y = data[4].values
X = data.drop(4, axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

train_data = Dataset(X_train, y_train)
eval_data = Dataset(X_test, y_test)

params = {}

print('Start training...')
gbt = GBT()
gbt.train(params,
          train_data,
          num_boost_round=20,
          valid_set=eval_data,
          early_stopping_rounds=10,
          objective="binary")

print('Start predicting...')
y_pred = []
for x in X_test:
    y_pred.append(gbt.predict(x, num_iteration=gbt.best_iteration))

print('The LogLoss of prediction is:', log_loss(y_test, inverse_logit_function(np.array(y_pred))))
print('The AUC of prediction is:', roc_auc_score(y_test, inverse_logit_function(np.array(y_pred))))

print('Load data...')
df_train = pd.read_csv('./data/regression.train', header=None, sep='\t')
df_test = pd.read_csv('./data/regression.test', header=None, sep='\t')

y_train = df_train[0].values
y_test = df_test[0].values
X_train = df_train.drop(0, axis=1).values
X_test = df_test.drop(0, axis=1).values

train_data = Dataset(X_train, y_train)
eval_data = Dataset(X_test, y_test)

params = {}

print('Start training...')
gbt = GBT()
gbt.train(params,
          train_data,
          num_boost_round=30,
          valid_set=eval_data,
          early_stopping_rounds=5)

print('Start predicting...')
y_pred = []
for x in X_test:
    y_pred.append(gbt.predict(x, num_iteration=gbt.best_iteration))

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
