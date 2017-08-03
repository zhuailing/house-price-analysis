#!/usr/bin/env python
# coding: utf-8 

#****************************************************************************************************
# 1.    Data Exploration and preprocessing:
# 2.    Regression modeling
# Please use code flat-dataExp.py for data exploration-c. univariate analysis and bivariate analysis
#****************************************************************************************************
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st
import xgboost as xgb
import sys
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV,LassoCV 
from sklearn.model_selection import cross_val_score
#%matplotlib inline

#****************************************************************************************************
# tools 
#****************************************************************************************************
# functions to compute evaluation score RMSE
def rmse(y_actual, y_predicted,modelname='Lasso'):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    print modelname +" RMSE= ",rmse
    return rmse

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
#****************************************************************************************************
# func to plot residuals
#****************************************************************************************************
def residual_plot(y_actual, y_predicted,modelname='Lasso'):
    matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
    preds = pd.DataFrame({"preds":y_predicted, "true":y_actual})
    preds["residuals"] = preds["true"] - preds["preds"]
    preds.plot(x = "preds", y = "residuals",kind = "scatter")
    plt.title("test residual plot-the" + modelname + " Model")
    plt.savefig(modelname + "_residual.png")
#****************************************************************************************************
# func to plot boxplot 
#****************************************************************************************************
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

#****************************************************************************************************
# 1.    Data Exploration and preprocessing:
# 2.    Regression modeling
#****************************************************************************************************
# 1.    Data Exploration and preprocessing:
#   a. check and remove missing value,noise
#   b. create new features and drop unwanted features
#   c. univariate analysis and bivariate analysis
#   d.  Normalize numeric features: transform the skewed numeric features by taking log(feature + 1) 
#   e.  Create Dummy variables for the categorical features
#   f.  split data into training and test
#****************************************************************************************************
#1  a. read data and check the data 
#
# There are 100331 records , 9 attributes and resale_price
#Index([u'month', u'town', u'flat_type', u'block', u'street_name',
#       u'storey_range', u'floor_area_sqm', u'flat_model',
#       u'lease_commence_date', u'resale_price'],
#      dtype='object')
#(100331, 10)
#****************************************************************************************************
data=pd.read_csv('resale-flat-prices-2012.csv')

#check missing values and remove noise
#(100330, 10)
#****************************************************************************************************
#check missing values
missing = data.isnull()

#remove noise
data=data[data.flat_model!='2-room']
print data.shape
#****************************************************************************************************
#   b. create new features and drop unwanted features
#
# add feature age_at_sale, year, month, price_per_sqm
# drop features = {'block', 'street_name'}
#(100330, 14)
#****************************************************************************************************
data=data.assign(price_per_sqm=data['resale_price']/data['floor_area_sqm'])
data=data.rename(columns={'month':'year-month'})
data['year'], data['month'] = data['year-month'].str.split('-', 1).str
data['year']=data['year'].astype(int)
data['month']=data['month'].astype(int)
data=data.assign(age_at_sale=data['year']-data['lease_commence_date'])

#change date to categorical data
data['year']=data['year'].astype(object)
data['month']=data['month'].astype(object)
data['lease_commence_date']=data['lease_commence_date'].astype(object)

print data.columns
#Index([u'year-month', u'town', u'flat_type', u'block', u'street_name',
#       u'storey_range', u'floor_area_sqm', u'flat_model',
#       u'lease_commence_date', u'resale_price', u'price_per_sqm', u'year',
#       u'month', u'age_at_sale'],
#      dtype='object')

#quantitative:['floor_area_sqm',  'resale_price', 'price_per_sqm', 'age_at_sale']
quantitative = [f for f in data.columns if data.dtypes[f] != 'object']
unwanted={'resale_price','price_per_sqm'}
quantitative = [e for e in quantitative if e not in unwanted]

#qualitative ['year-month','lease_commence_date', 'town', 'flat_type', 'block', 'street_name', 'storey_range','year','month', 'flat_model']
qualitative = [f for f in data.columns if data.dtypes[f] == 'object']
unwanted = {'block', 'street_name'}
qualitative = [e for e in qualitative if e not in unwanted]

#****************************************************************************************************
#final data include: 2+8 =10 features, 1 target
#quantitative=['floor_area_sqm', 'age_at_sale']
#qualitative=['year-month','lease_commence_date','year','month', 'town', 'flat_type',  'storey_range', 'flat_model']
#target=['resale_price']
#****************************************************************************************************
#   d.  Normalize numeric features: transform the skewed numeric features by taking log(feature + 1) 
#   e.  Create Dummy variables for the categorical features
#(100330, 212) 211 features
#****************************************************************************************************
all_feats=quantitative + qualitative + ['resale_price']
data=data[all_feats]

#log transform skewed numeric features:
numeric_feats=['floor_area_sqm', 'age_at_sale','resale_price']
data[numeric_feats] = np.log1p(data[numeric_feats])

#Create Dummy variables for the categorical features
data = pd.get_dummies(data)
print data.shape
#Index([u'floor_area_sqm', u'age_at_sale', u'resale_price',
#    ...
#       u'flat_model_Multi Generation', u'flat_model_New Generation',
#       u'flat_model_Premium Apartment', u'flat_model_Premium Apartment Loft',
#       u'flat_model_Premium Maisonette', u'flat_model_Simplified',
#       u'flat_model_Standard', u'flat_model_Terrace', u'flat_model_Type S1',
#       u'flat_model_Type S2'],
#      dtype='object', length=212)
#
#****************************************************************************************************
#   f.  split data into training and test
#split data into training(2012-2015) and test (year: 2016+2017)
#test: 2016-19379, 2017-9715 total (29094, 212)
#train: 2012-1015: total (71236, 212)
# total: (100330, 212)
#****************************************************************************************************
test=data.query('year_2017 == 1 | year_2016 == 1')
train=data.query('year_2017 == 0 & year_2016 == 0')
y_train=train.pop('resale_price')
y_test=test.pop('resale_price')
x_train=train
x_test=test
print x_train.shape
print x_test.shape
#****************************************************************************************************
# 2.    Regression modeling
#   a.  Ridge 
#   b.  Lasso 
#   c.  Random Forest
#   d.  XGboost
#   e. Scatter plot: XGBoost vs Ridge
#****************************************************************************************************
#****************************************************************************************************
#   a.  Ridge 
#tuning parameter alpha using cross validation
#****************************************************************************************************
def create_Ridge():
    model_ridge = Ridge()
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
    cv_ridge = pd.Series(cv_ridge, index = alphas)
    cv_ridge.plot(title = "Ridge - alpha tuning using 5 fold-CV")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    plt.savefig("Ridge_alpha.png")

    #best alpha=30

    cv_ridge.min()
    print cv_ridge.min()
    #0.097343346574955442

    ridge_model = Ridge(alpha=30)
    ridge_model.fit(x_train,y_train)

    y_test_pred = ridge_model.predict(x_test)
    rmse_test_ridge=rmse(y_test, y_test_pred, 'Ridge')
    # RMSE=  0.118403910858

    residual_plot(y_test, y_test_pred,modelname='Ridge')
    return y_test_pred
#****************************************************************************************************
#   b.  Lasso 
#   use Lasso CV to figure out the best alpha 
#   the alphas in Lasso CV are really the inverse or the alphas in Ridge.
#   check the feature importance from coefficients
#****************************************************************************************************
def create_Lasso():

    lasso_model= LassoCV(alphas = [1, 0.1,0.03, 0.001, 0.0005]).fit(x_train, y_train)
    rmse_cv(lasso_model).mean()
    print rmse_cv(lasso_model).mean()
    #0.10114804241488809
    # >>> lasso_model.alpha_= 0.00050000000000000001

    y_test_pred = lasso_model.predict(x_test)
    rmse_test_Lasso=rmse(y_test, y_test_pred,'Lasso')
    #Lasso RMSE=  0.12658011435
    

    #   check the feature importance from coefficients
    coef = pd.Series(lasso_model.coef_, index = x_train.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    #Lasso picked 68 variables and eliminated the other 143 variables

    imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")
    plt.tight_layout()
    plt.savefig("Lasso_featImp.png")

    #let's look at the residuals as well:
    #matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
    #preds = pd.DataFrame({"preds":model_lasso.predict(x_test), "true":y_test})
    #preds["residuals"] = preds["true"] - preds["preds"]
    #preds.plot(x = "preds", y = "residuals",kind = "scatter")
    #plt.title("test residual plot-the Lasso Model")

    residual_plot(y_test, y_test_pred,modelname='Lasso')
#****************************************************************************************************
#   c.  Random Forest
#   use GridSearchCV find the best parameters
#   extract feature importance from RF model
#****************************************************************************************************
def create_RF():

    n_estimators=200
    random_state=0
    n_jobs=-1
    #min_samples_leaf = 50

    ## Create a Random Forest regressor object
    rf = RandomForestRegressor(n_estimators=n_estimators,oob_score = True,random_state=random_state,n_jobs=n_jobs)

    param_grid={"max_features": [12,14,16,18]}
    #param_grid={"max_features": [14]}
    #param_grid={"max_features": [7,8,9,10,12,14,15,17],"n_estimators":[600,800,1000]}
    #param_grid={"n_estimators":[200,400,600,800,1000]}
#    grid_search = GridSearchCV(rf,cv=3, scoring='neg_mean_squared_error', param_grid=param_grid)
#    grid_search.fit(x_train,y_train)
#
#    print "the best parameter:",grid_search.best_params_
#    print "the best score:",grid_search.best_score_
#    #the best parameter: {'max_features':14,'n_estimators': 1000}
#
#
#    y_test_pred= grid_search.predict(x_test)
#    rmse_test_RF=rmse(y_test, y_test_pred,"RF")


    rf_best = RandomForestRegressor(n_estimators=1000,oob_score = True,max_features=14,random_state=random_state,n_jobs=n_jobs)
    rf_best.fit(x_train,y_train)

    y_test_pred= rf_best.predict(x_test)
    rmse_test_RF=rmse(y_test, y_test_pred,"RF")
    #RF RMSE=  0.121871804159
    residual_plot(y_test, y_test_pred,modelname='RF')
    

    # get the feature importance
    importances=rf_best.feature_importances_ 
    # print("Original ",np.argsort(importances))
    feat_labels= x_train.columns

    indices = np.argsort(importances)[::-1]
    # print (" importances ",importances)
    # print (" indices ",indices)

    # write the feature importance to file
    sys.stdout=open("featImpRF.txt","w")
    for f in range(x_train.shape[1]):
        print("%2d\t%-*s\t%f" % (f+1,30,feat_labels[indices[f]], importances[indices[f]]))

    sys.stdout.close()
    sys.stdout=sys.__stdout__

    # Plot the feature importance in the bar chart.
    importance_frame = pd.DataFrame({'Importance': importances[indices[:20]], 'Feature':feat_labels[indices[:20]]})
    importance_frame.sort_values(by = 'Importance', inplace = True)
    importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (14,8), color = 'orange',align='center')
    plt.tight_layout()
    plt.title("Feature Importance -RF")
    plt.savefig("FeatImpRF.png")

#****************************************************************************************************
#   d. XGBoost
#       use GridSearchCV to tune parameters
#****************************************************************************************************
def create_XGboost():
    param_dist = { "n_estimators":1000, \
                    "learning_rate":0.1,\
                    "max_depth":2\
                    }
    #xgb_model = xgb.XGBRegressor(n_estimators=1000, max_depth=2, learning_rate=0.1) 
    xgb_model = xgb.XGBRegressor(param_dist)

    param_test = {'max_depth':[2,4,6], \
                    'min_child_weight':[6], \
                    'n_estimators': [100,500,1000],\
                    'gamma':[0,0.1], \
                    'reg_alpha':[0.001], \
                    'learning_rate':[0.1]}

    #param_test = {'max_depth':[2,4,6]}
#    param_test = {'max_depth':[6]}
#    grid_search = GridSearchCV(xgb_model,cv=3, scoring='neg_mean_squared_error', param_grid=param_test)
#    grid_search.fit(x_train,y_train)
#    print "the best parameter:",grid_search.best_params_
#    #the best parameter: {'max_depth': 6}
#
#
#    y_test_pred= grid_search.predict(x_test)
#    rmse_test_RF=rmse(y_test, y_test_pred,"XGboost")
#    residual_plot(y_test, y_test_pred,'XGboost')

    #the params were tuned using GridSearchCV
    xgb_model = xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.1) 
    xgb_model.fit(x_train,y_train)
    y_test_pred= xgb_model.predict(x_test)
    rmse_test_RF=rmse(y_test, y_test_pred,"XGboost")
    #XGboost RMSE=  0.11511042121
    residual_plot(y_test, y_test_pred,'XGboost')
    return y_test_pred

#****************************************************************************************************
#   e. Scatter plot: XGBoost vs Ridge
#****************************************************************************************************
def scatter_plot(xgb_preds,ridge_preds):

    #xgb_preds = np.expm1(xgb_model.predict(x_test))
    #ridge_preds = np.expm1(ridge_model.predict(x_test))
    xgb_preds = np.expm1(xgb_preds)
    ridge_preds = np.expm1(ridge_preds)
    predictions = pd.DataFrame({"xgb":xgb_preds, "ridge":ridge_preds})
    predictions.plot(x = "xgb", y = "ridge", kind = "scatter")
    plt.title("Predictions: XGboost vs Ridge ")
    plt.savefig("scatterPlot_XGboost-Ridge.png")
#****************************************************************************************************
# main function
#****************************************************************************************************
def main():
    ridge_preds=create_Ridge()
    #print ridge_preds.shape
    create_Lasso()
    create_RF()
    xgb_preds=create_XGboost()
    scatter_plot(xgb_preds,ridge_preds)
    print "me"

if __name__=="__main__":
    main()
