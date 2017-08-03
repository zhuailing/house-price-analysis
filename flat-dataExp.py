#!/usr/bin/env python
# coding: utf-8 

#****************************************************************************************************
# there are two parts: 
# 1.    Data Exploration and preprocessing:
# 2.    Regression modeling
# This is part 1:  Data Exploration 
#   c. univariate analysis and bivariate analysis
# Please use code flat-model.py for data preprocessing and regression modeling
#****************************************************************************************************
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st
from math import sqrt
from sklearn.preprocessing import StandardScaler
#%matplotlib inline
#****************************************************************************************************
# func to plot boxplot 
#****************************************************************************************************
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
#****************************************************************************************************
# 1.    Data Exploration and preprocessing:
#   a. check and remove missing value,noise
#   b. create new features and drop unwanted features
#   c. univariate analysis and bivariate analysis
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
print data.describe()
print data.shape
print data.dtypes
print data.columns

#check missing values and remove noise
#****************************************************************************************************
#check missing values
missing = data.isnull()

#remove noise
data=data[data.flat_model!='2-room']
#data[data.flat_model!='2-room'].shape
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

data.columns
#Index([u'year-month', u'town', u'flat_type', u'block', u'street_name',
#       u'storey_range', u'floor_area_sqm', u'flat_model',
#       u'lease_commence_date', u'resale_price', u'price_per_sqm', u'year',
#       u'month', u'age_at_sale'],
#      dtype='object')

#quantitative:['floor_area_sqm',  'resale_price', 'price_per_sqm', 'age_at_sale']
quantitative = [f for f in data.columns if data.dtypes[f] != 'object']
unwanted={'resale_price','price_per_sqm'}
quantitative = [e for e in quantitative if e not in unwanted]
#quantitative=['floor_area_sqm', 'age_at_sale']

#qualitative ['year-month','lease_commence_date', 'town', 'flat_type', 'block', 'street_name', 'storey_range','year','month', 'flat_model']
qualitative = [f for f in data.columns if data.dtypes[f] == 'object']
unwanted = {'block', 'street_name'}
qualitative = [e for e in qualitative if e not in unwanted]
#qualitative=['year-month','lease_commence_date','year','month', 'town', 'flat_type',  'storey_range', 'flat_model']
#****************************************************************************************************
# you can look at Summary of numerical fields by using describe() function
#****************************************************************************************************
def summary_data():

    temp=data.describe(include='all')
    temp=data.describe(include=['number'])
    temp=data.describe(include=['object'])
    temp=data.describe()
    filename='data-summary.csv'
    temp.to_csv(filename)

    for col in data.select_dtypes(include=['object']):
        temp = data[col].value_counts()
        temp.to_csv(filename, header=False, mode = 'a')
    temp.to_csv(filename, header=False, mode = 'a')


#****************************************************************************************************
#   c. univariate analysis and bivariate analysis
#****************************************************************************************************
#check the distribution of quantitative features and resale_price:
# and Test normality 
#****************************************************************************************************
def univar_analysis():
    y = data['resale_price']
    plt.figure(1); plt.title('Johnson SU')
    sns.distplot(y, kde=False, fit=st.johnsonsu)
    plt.figure(2); plt.title('Normal')
    sns.distplot(y, kde=False, fit=st.norm)
    plt.figure(3); plt.title('Log Normal')
    sns.distplot(y, kde=False, fit=st.lognorm)
    plt.savefig("HistLogNorm.png")
    #plt.ion()
    #plt.show()

    #skewness and kurtosis
    print("Skewness: %f" % data['resale_price'].skew())
    print("Kurtosis: %f" % data['resale_price'].kurt())
    # Skewness: 1.465601
    # Kurtosis: 3.005986
    #positive kurtosis indicates a "heavy-tailed" distribution

    #quantitative:['floor_area_sqm', 'age_at_sale']
    f = pd.melt(data, value_vars=quantitative)
    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")
    plt.savefig("HistAll.png")

    #Test normality: 
    test_normality = lambda x: st.shapiro(x)[1] < 0.01
    normal = pd.DataFrame(data[quantitative])
    normal = normal.apply(test_normality)
    print(not normal.any())
    #False
#****************************************************************************************************
#Univariate analysis: Check outliers for resale_price
#****************************************************************************************************
#The primary concern here is to establish a threshold that defines an observation as an outlier. 
#To do so, we'll standardize the data.
#****************************************************************************************************
def check_outlier():
    #standardizing data
    resale_price_scaled = StandardScaler().fit_transform(data['resale_price'][:,np.newaxis]);
    low_range = resale_price_scaled[resale_price_scaled[:,0].argsort()][:10]
    high_range= resale_price_scaled[resale_price_scaled[:,0].argsort()][-10:]
    print('outer range (low) of the distribution:')
    print(low_range)
    print('\nouter range (high) of the distribution:')
    print(high_range)
    #outer range (low) of the distribution:
    #[[-1.99006788]
    # [-1.97476194]
    # [-1.95180304]
    # [-1.95180304]
    # [-1.92884413]
    # [-1.91353819]
    # [-1.91353819]
    # [-1.91353819]
    # [-1.91353819]
    # [-1.91353819]]
    #outer range (high) of the distribution:
    #[[ 4.88229866]
    # [ 4.8890945 ]
    # [ 4.97413429]
    # [ 4.97413429]
    # [ 4.97413429]
    # [ 5.03535805]
    # [ 5.0965818 ]
    # [ 5.12719368]
    # [ 5.35678276]
    # [ 5.58637185]]
    #Low range values are similar and not too far from 0.
#High range values are far from 0 and the 5.something values are really out of range.
#For now, we'll not consider any of these values as an outlier but we should be careful with those 6 5.something values.
#****************************************************************************************************
#Bivariate analysis: 
#   Relationship with numerical variables
#   Relationship with Categorical variables
#****************************************************************************************************
def bivar_analysis():
    #****************************************************************************************************
    #Relationship with numerical variables
    #scatter plot saleprice with each quantitive feature
    #quantitative=['floor_area_sqm', 'age_at_sale']
    #****************************************************************************************************
    f = pd.melt(data, id_vars=['resale_price'], value_vars=quantitative)
    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
    g = g.map(plt.scatter, "value", "resale_price")
    plt.savefig("Scatter.png")

    #They look like special cases, however they seem to be following the trend. For that reason, we will keep them.
    #****************************************************************************************************
    #Relationship with Categorical variables
    #check distribution of SalePrice with respect to variable values
    #qualitative=['year-month','lease_commence_date','year','month', 'town', 'flat_type',  'storey_range', 'flat_model']
    #****************************************************************************************************
    f = pd.melt(data, id_vars=['resale_price'], value_vars=qualitative)
    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
    g = g.map(boxplot, "value", "resale_price")
    plt.savefig("BoxPlotResale.png")

#****************************************************************************************************
# main function
#****************************************************************************************************
def main():
    summary_data()
    univar_analysis()
    check_outlier()
    bivar_analysis()

if __name__=="__main__":
    main()
