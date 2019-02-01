
import numpy as np

import pandas as pd



from scipy.stats import skew

from scipy.special import boxcox1p

from sklearn.feature_selection import RFECV
from sklearn import  linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
import lightgbm as lgb


train = pd.read_csv('data/train.csv')

test = pd.read_csv('data/test.csv')

#train2 = pd.read_csv('pytrain.csv')

#test2 = pd.read_csv('pytest.csv')

# remove outliers

train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000))]



all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))



# drop some features to avoid multicollinearity

train["SalePrice"] = np.log1p(train["SalePrice"])

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.65]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = boxcox1p(all_data[skewed_feats], 0.15)

all_data = pd.get_dummies(all_data)

all_data["TotalSF"]=all_data["GrLivArea"]+all_data["1stFlrSF"]+all_data["2ndFlrSF"]


all_data = all_data.fillna(all_data.mean())

all_data.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd', '2ndFlrSF'], axis=1, inplace=True)


X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice

#y2 = train2.SalePrice

#train2=train2.drop('SalePrice',1)
#test2=test2.drop('SalePrice',1)

#### models selection

lasso = Lasso(alpha=0.0004)
lasso2 = Lasso(alpha=0.0004)


# fit model on training data
XGBmodel = XGBClassifier()
lassomodel = lasso
lassomodel2= lasso2
myGBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.02,
                                      max_depth=4, max_features='sqrt',
                                      min_samples_leaf=15, min_samples_split=50,
                                      loss='huber', random_state = 5) 
                                      
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=4.0, l1_ratio=0.005, random_state=3))    
myLGB = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=600,
                              max_bin = 50, bagging_fraction = 0.6,
                              bagging_freq = 5, feature_fraction = 0.25,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)
l2Regr = Ridge(alpha=9.0, fit_intercept = True)
#lm = linear_model.LinearRegression()

### prediction

#lassomodel.fit(train2, y2)
#lassomodel2.fit(train2, y2)
#myGBR.fit(train2,y2)
#ENet.fit(train2,y2)
#myLGB.fit(train2,y2)
#l2Regr.fit(train2,y2)
#lm.fit(train2,y2)
lassomodel.fit(X_train, y)
#XGBmodel.fit(X_train, y)
myGBR.fit(X_train,y)
#ENet.fit(X_train,y)
myLGB.fit(X_train,y)
l2Regr.fit(X_train,y)
#lm.fit(X_train,y)


#Lasso2 = np.expm1(lassomodel2.predict(test2))
#mygbr = np.expm1(myGBR.predict(test2))
#net   = np.expm1(ENet.predict(test2))
#mylgb = np.expm1(myLGB.predict(test2))
#ridge = np.expm1(l2Regr.predict(test2))
#Lm    = np.expm1(lm.predict(test2))

Lasso = np.expm1(lassomodel.predict(X_test))
mygbr = np.expm1(myGBR.predict(X_test))
#net   = np.expm1(ENet.predict(X_test))
mylgb = np.expm1(myLGB.predict(X_test))
ridge = np.expm1(l2Regr.predict(X_test))
#Lm    = np.expm1(lm.predict(X_test))
res=(2*Lasso+2*ridge+1.5*mygbr+0.5*mylgb)/6
solution = pd.DataFrame({"id":test.Id, "SalePrice":res})
solution.to_csv("franco.csv", index = False)



