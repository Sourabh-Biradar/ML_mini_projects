# %%
# Random Forest : RF

# RF-Regressor

# carsdekho.com dataset

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('cardekho.csv')

df.drop(['Unnamed: 0','car_name','brand'],axis=1,inplace=True)

df.head()

# %%
df.isnull().sum()

# %%
df.info()

# %%
# separating numeric features -(discrete , continuous) , categorical features 

num_cols = df.select_dtypes(exclude='object').columns
cat_cols = df.select_dtypes(include='object').columns

dis_feats = [feat for feat in num_cols if (len(df[feat].unique()))>=25]
cont_feats = [feat for feat in num_cols if feat not in dis_feats]

# %%
# labesls & target
X = df.drop('selling_price',axis=1)
y = df['selling_price']

# %%
# Encodind : Label & OneHot

len(df['model'].unique())

# %%
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

le = LabelEncoder()
X['model'] = le.fit_transform(X['model'])

oh_cols = ['seller_type','fuel_type','transmission_type']

oh = OneHotEncoder()

preprocessor = ColumnTransformer(
    [('OneHotEn',oh,oh_cols)],
    remainder = 'passthrough'
)

X = preprocessor.fit_transform(X)

# %%
# train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

# %%
# model fitting & training

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# %%
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def evaluate_models(true_vals,pred_vals):
    mae = mean_absolute_error(true_vals,pred_vals)
    mse = mean_squared_error(true_vals,pred_vals)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_vals,pred_vals)

    return mae,rmse,r2

# %%
models = {
    "Linear Regression":LinearRegression(),
    'Lasso':Lasso(),
    'Ridge':Ridge(),
    'KNN-Reg':KNeighborsRegressor(),
    'DT-Reg' :DecisionTreeRegressor(),
    'RF-Reg' :RandomForestRegressor()
}

for name,model in models.items():
    model.fit(X_train,y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_mae , model_train_rmse, model_train_r2 = evaluate_models(y_train,y_train_pred)
    model_test_mae , model_test_rmse, model_test_r2 = evaluate_models(y_test,y_test_pred)

    print(f'------model performance of {name} for training set-------')
    print(f' mae = {model_train_mae:.3f}')
    print(f' rmse = {model_train_rmse:.3f}')
    print(f' r2_score = {model_train_r2:.3f}')
    print('-'*15)
    print(f'------model performance of {name} for test set-------')
    print(f' mae = {model_test_mae:.3f}')
    print(f' rmse = {model_test_rmse:.3f}')
    print(f' r2_score = {model_test_r2:.3f}')

# %%
# selecting best performing model & hyperparameter tunning them to get even better performance

dtr_params = {'max_depth':[1,2,5,10],'splitter':['best','random']}

rfr_params = {
    'max_depth':[5,8,15,None,10],
    'max_features':[5,7,'auto',8],
    'min_samples_split':[2,8,15,20],
    'n_estimators':[100,200,500,1000]
}

randomcv_models = [
    ('DF-Reg',DecisionTreeRegressor(),dtr_params),
    ('RF-Reg',RandomForestRegressor(),rfr_params)
]

# %%
# Hyperparam tunning

import warnings;warnings.filterwarnings('ignore')

from sklearn.model_selection import RandomizedSearchCV

model_params = {}

for name,model,params in randomcv_models:
    random = RandomizedSearchCV(model,
                                param_distributions=params,
                                cv=3)

    random.fit(X_train,y_train)

    model_params[name]=random.best_params_

for model_name in model_params:
    print(f'------ Best Params for {model_name}------')
    print(model_params[model_name])

# %%
# substituting best params back to models

models = {
    'DT-Reg':DecisionTreeRegressor(splitter='best' ,max_depth =10),
    'RF-Reg':RandomForestRegressor(n_estimators=200, min_samples_split=8, max_features=8, max_depth=None)
}

for name,model in models.items():
    model.fit(X_train,y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred=model.predict(X_test)

    model_train_mae , model_train_rmse, model_train_r2 = evaluate_models(y_train,y_train_pred)
    model_test_mae , model_test_rmse, model_test_r2 = evaluate_models(y_test,y_test_pred)

    print(f'------model performance of {name} for training set-------')
    print(f' mae = {model_train_mae:.3f}')
    print(f' rmse = {model_train_rmse:.3f}')
    print(f' r2_score = {model_train_r2:.3f}')
    print('-'*15)
    print(f'------model performance of {name} for test set-------')
    print(f' mae = {model_test_mae:.3f}')
    print(f' rmse = {model_test_rmse:.3f}')
    print(f' r2_score = {model_test_r2:.3f}')


