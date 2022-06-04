#전처리 및 학습

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings(action='ignore')

train = pd.read_csv('enco_train_lunch.csv')
test = pd.read_csv('enco_test_lunch.csv')

x_train = train[['day', 'numbers', 'dayoff', 'work', 'outsidework', 'workfhome','Month','Date','bob','soup','main']]
y_train = train['lunch_t'] 
x_test = test[['day', 'numbers', 'dayoff', 'work', 'outsidework', 'workfhome','Month','Date','bob','soup','main']]

from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBRegressor

# param = {
#     'max_depth':[2,3,4],
#     'n_estimators':range(300,600,100), #  'n_estimators':range(600,700,50) 여기에 cv 10 (이거와 별반차이가 없다.)
#     'colsample_bytree':[0.5,0.7,1],
#     'colsample_bylevel':[0.5,0.7,1],
# }
# model = xgb.XGBRegressor()
# grid_search = GridSearchCV(estimator=model, param_grid=param, cv=10, 
#                            scoring='neg_mean_squared_error',
#                            n_jobs=-1)

# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)
{'colsample_bylevel': 0.5, 'colsample_bytree': 0.5, 'max_depth': 3, 'n_estimators': 600}
xgb_model = XGBRegressor(booster='gbtree', colsample_bylevel=0.7593,
             colsample_bytree=0.5391, gamma=1.235, learning_rate=0.017,
             max_depth=5, min_child_weight=1, missing=None, n_estimators=364,
             n_jobs=1, subsample=1)

xgb_model.fit(x_train, y_train)

y_pred2 = xgb_model.predict(x_test)

train = pd.read_csv('enco_train_dinner.csv')
test = pd.read_csv('enco_test_dinner.csv')

x_train = train[['day', 'numbers', 'dayoff', 'work', 'outsidework', 'workfhome','Month','Date','bobd','soupd','maind']]
y_train = train['dinner_t'] 

x_test = test[['day', 'numbers', 'dayoff', 'work', 'outsidework', 'workfhome','Month','Date','bobd','soupd','maind']]

from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBRegressor

# param = {
#     'max_depth':[2,3,4],
#     'n_estimators':range(300,600,100), #  'n_estimators':range(600,700,50) 여기에 cv 10 (이거와 별반차이가 없다.)
#     'colsample_bytree':[0.5,0.7,1],
#     'colsample_bylevel':[0.5,0.7,1],
# }
# model = xgb.XGBRegressor()
# grid_search = GridSearchCV(estimator=model, param_grid=param, cv=10, 
#                            scoring='neg_mean_squared_error',
#                            n_jobs=-1)

# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)
# {'colsample_bylevel': 0.5, 'colsample_bytree': 0.5, 'max_depth': 3, 'n_estimators': 600}
xgb_model = XGBRegressor(booster='gbtree', colsample_bylevel=0.5551,
             colsample_bytree=0.5551, gamma=1.6727, learning_rate=0.0175,
             max_depth=5, min_child_weight=1, missing=None, n_estimators=565,
             n_jobs=1,subsample=0.8710)

xgb_model.fit(x_train, y_train)

y_pred_dinner2 = xgb_model.predict(x_test)
submit = pd.read_csv('원본/sample_submission.csv')
submit['중식계'] = y_pred2
submit['석식계'] = y_pred_dinner2
submit
submit.to_csv('final_end1.csv', index=False)
