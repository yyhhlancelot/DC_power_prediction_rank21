import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.tree import export_graphviz
import graphviz
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn import manifold, decomposition, ensemble, discriminant_analysis, random_projection
from time import time
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

from matplotlib import offsetbox
from time import sleep
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

'''
国能日新 功率预测题专用function
editor : yyh
'''

def data_selection(train_data, train_list_i):
    '''
    选择与训练集相似的时间进行训练
    editor : yyh
    '''
    print(train_data)
    os.system("pause")
    train_data['month'] = train_data['时间'].apply(lambda x: x[5:7]).astype('int32')
    
    if (train_list_i == 'train_1.csv'):
        train_data = train_data[(train_data['month'].isin([5]) | train_data['month'].isin([6]) | train_data['month'].isin([7]) | train_data['month'].isin([8]))]
    if (train_list_i == 'train_2.csv'):
        train_data = train_data[(train_data['month'].isin([5]) | train_data['month'].isin([6]) | train_data['month'].isin([7]) | train_data['month'].isin([8]) | train_data['month'].isin([9]))]
    if (train_list_i == 'train_3.csv'):
        train_data = train_data[(train_data['month'].isin([8]) | train_data['month'].isin([9]) | train_data['month'].isin([10]))]
    if (train_list_i == 'train_4.csv'):
        train_data = train_data[(train_data['month'].isin([5]) | train_data['month'].isin([6]) | train_data['month'].isin([7]) | train_data['month'].isin([8]) | train_data['month'].isin([9]))]
        
    train_data = train_data.reset_index(drop = True)
    
    print(train_data)
    os.system("pause")
    return train_data

def plot_embedding(X, y, title=None):
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min) # 已经归一化
    plt.figure()
    ax = plt.subplot(111)
    print(X.shape[0])
    os.system("pause")
    print(X)
    os.system("pause")
    print(X[0, 0], X[0, 1])
    ax.scatter(X[:, 0], X[:, 1])
    plt.show()
    os.system("pause")
    for i in range(X.shape[0]):
        shown_points = np.array([[1., 1.]])
        
        dist = np.sum((X[i] - shown_points) ** 2, 1) # a^2 + b^2 
        
        # if np.min(dist) < 1:
            # continue
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
            
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def pca(X):
    """
    pca analysis written and integrated in a function
    editor : yyh
    """
    print("computing pca projection")
    t0 = time()
    pca = decomposition.TruncatedSVD(n_components = 2)
    X_pca = pca.fit_transform(X)
    
    print("time of pca used is %.2fs" % (time() - t0))
    # plt.show()
    
    return X_pca
    
def t_sne(X):
    """
    t_sne analysis written and integrated in a function
    editor : yyh
    """
    print("computing t-SNE projection")
    tsne = manifold.TSNE(n_components = 2, init = 'pca', random_state = 0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    # os.system("pause")
    # plot_embedding(X_tsne, y, "t-SNE embedding of the features (time %.2fs)" % (time() - t0))
    # plt.show()
    print("time of tsne used is %.2fmin %.2fs" % ((time() - t0) // 60, (time() - t0) % 60))
    
    return X_tsne

def normalize(d_column):
    """z-score normalization
       the feature will obey the (0, 1) normal distribution
       editor : yyh
       """
       
    d_column_mean = d_column.mean()
    
    d_column_std = d_column.std()
    
    d_column_new = (d_column - d_column_mean) / d_column_std
    
    return d_column_new

def normalize_all(dataframe):
    """
    normalize dataframe which has no string element
    editor : yyh
    """
    for i in dataframe.columns:
        dataframe[i] = normalize(dataframe[i])
    return dataframe
    
def get_month(x, data_list_i):
    """process the first wrong month"""
    
    year = int(x[0 : 4])
    mon = int(x[5 : 7])
    day = int(x[8 : 10])
    h = int(x[11 : 13])
    m = int(x[14 : 16])
    
    if (data_list_i == 'test_1.csv') & (year == 2018) & (mon == 4) & (day == 30) & (h == 23) & (m == 59):
        return 5
    
    if (data_list_i == 'test_2.csv') & (year == 2018) & (mon == 4) & (day == 30) & (h == 23) & (m == 59):
        return 5
    
    if (data_list_i == 'test_3.csv') & (year == 2018) & (mon == 7) & (day == 31) & (h == 23) & (m == 59):
        return 8
    
    
    return mon

def get_day(x, data_list_i):
    """process the first wrong day"""
    year = int(x[0 : 4])
    mon = int(x[5 : 7])
    day = int(x[8 : 10])
    h = int(x[11 : 13])
    m = int(x[14 : 16])
    
    if (data_list_i == 'train_2.csv') & (year == 2017) & (mon == 1) & (day == 1) & (h == 23) & (m == 59):
        return day + 1
        
    if (data_list_i == 'train_4.csv') & (year == 2017) & (mon == 1) & (day == 1) & (h == 23) & (m == 59):
        return day + 1
    
    if (data_list_i == 'test_1.csv') & (year == 2018) & (mon == 4) & (day == 30) & (h == 23) & (m == 59):
        return 1
        
    if (data_list_i == 'test_2.csv') & (year == 2018) & (mon == 4) & (day == 30) & (h == 23) & (m == 59):
        return 1
    
    if (data_list_i == 'test_3.csv') & (year == 2018) & (mon == 7) & (day == 31) & (h == 23) & (m == 59):
        return 1
        
    if (data_list_i == 'test_3.csv') & (year == 2018) & (mon == 9) & (day == 26) & (h == 23) & (m == 59):
        return 28
    
    return day


def get_time(x):
    h = int(x[11:13])
    m = int(x[14:16])
    if m in [14, 29, 44]:
        m += 1
    if m == 59:
        m = 0
        h += 1
    if h == 24:
        h = 0
    return h * 60 + m
    
def get_hour(x):
    h = int(x[11:13])
    m = int(x[14:16])
    if m in [14, 29, 44]:
        m += 1
    if m == 59:
        m = 0
        h += 1
    if h == 24:
        h = 0
    return h

def get_min(x):
    h = int(x[11:13])
    m = int(x[14:16])
    if m in [14, 29, 44]:
        m += 1
    if m == 59:
        m = 0
        h += 1
    if h == 24:
        h = 0
    return m
    
    
def add_poly_features(data, column_names):
    """
    polynomia features
    """
    
    
    features = data[column_names]
    rest_features = data.drop(column_names, axis=1)
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = pd.DataFrame(poly_transformer.fit_transform(features),
                                 columns=poly_transformer.get_feature_names(column_names))
    
    
    for col in poly_features.columns:
        if col in rest_features.columns.tolist():
            continue
        rest_features.insert(1, col, poly_features[col])
    return rest_features

def add_plus_features(data, column_names):
    """
    make elements in column_names add each other and normalize
    editor : yyh
    """
    column_list = data.columns.tolist()
    
    for i in range(len(column_names)):
        for j in range(i + 1, len(column_names)):
            if (column_names[i] + '+' + column_names[j] in column_list):
                continue
            data[column_names[i] + '+' + column_names[j]] = normalize(data[column_names[i]] + data[column_names[j]])
    return data

def add_sub_features(data, column_names):
    """
    make elements in column_names minus each other and normalize
    editor : yyh
    """
    column_list = data.columns.tolist()
    
    for i in range(len(column_names)):
        for j in range(len(column_names)):
            if ((j == i) |(column_names[i] + '-' + column_names[j] in column_list)):
                continue
            data[column_names[i] + '-' + column_names[j]] = normalize(data[column_names[i]] - data[column_names[j]])
    return data
    
def add_div_features(data, column_names):
    """
    make elements in column_names divide each other and normalize
    editor : yyh
    """
    column_list = data.columns.tolist()
    
    for i in range(len(column_names)):
        for j in range(len(column_names)):
            if ((j == i)|(column_names[i] + '/' + column_names[j] in column_list)):
                continue
            data[column_names[i] + '/' + column_names[j]] = normalize(data[column_names[i]] / data[column_names[j]])
    
            
    return data
    
    
def dis2peak(feature_col, real_irradiance_col, year_col, month_col, day_col):
    """
    search the max real_irradiance with the time of everyday, compute the time distance with the max-irradiance-time.
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    while (index_new < len(feature_col)):
        
        # index
        
        a = real_irradiance_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        b = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        var_temp = a.max()
        
        var.loc[index_new : index_new + len(a) - 1, 'output'] = abs(b.loc[index_new : index_new + len(a) - 1] - b[(a == var_temp)].values[0])
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)
    
def numerical_4_feature(feature_col, year_col, month_col, day_col, irradiance_col, method = None, time_period = None):
    """
    numerical analysis for input feature of appointed period time 
    editor : yyh
    """
    if (time_period not in ['allday', 'daytime', 'nighttime']):
        
        print("time_period is not in the list!")
        
        os.system("pause")
    
    if (method not in ['mean', 'std', 'max', 'min', 'var']):
        
        print("method is not in the list!")
        
        os.system("pause")
        
    variety = pd.DataFrame()
    
    variety['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
        
    if time_period == 'allday':
        # index
        while (index_new < len(feature_col)):
        
            a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
            
            if (method == 'mean'):
                var_temp = a.mean()
            if (method == 'std'):
                var_temp = a.std()
            if (method == 'max'):
                var_temp = a.max()
            if (method == 'min'):
                var_temp = a.min()
            if (method == 'var'):
                var_temp = a.max() - a.min()
            # sleep(1)
            
            variety.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
            
            #update index
            index_new = index_new + len(a)
            
            if(index_new == len(feature_col)):
                break
            
            s_y = year_col.loc[index_new]
            s_m = month_col.loc[index_new]
            s_d = day_col.loc[index_new]
        
    if time_period == 'daytime':
        while (index_new < len(feature_col)):
        # index
            a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
            
            b = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (irradiance_col != -1)]
            
            # if len(a) != 1:
            if (method == 'mean'):
                var_temp = b.mean()
            if (method == 'std'):
                var_temp = b.std()
            if (method == 'max'):
                var_temp = b.max()
            if (method == 'min'):
                var_temp = b.min()
            if (method == 'var'):
                var_temp = b.max() - b.min()
            
            variety.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
            
            #update index
            index_new = index_new + len(a)
            
            if(index_new == len(feature_col)):
                break
            
            s_y = year_col.loc[index_new]
            s_m = month_col.loc[index_new]
            s_d = day_col.loc[index_new]
    
    if time_period == 'nighttime':
        while (index_new < len(feature_col)):
            # index
            a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
            
            b = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (irradiance_col == -1)]
            
            if (method == 'mean'):
                var_temp = b.mean()
            if (method == 'std'):
                var_temp = b.std()
            if (method == 'max'):
                var_temp = b.max()
            if (method == 'min'):
                var_temp = b.min()
            if (method == 'var'):
                var_temp = b.max() - b.min()
            # sleep(1)
            
            variety.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
            
            #update index
            index_new = index_new + len(a)
            
            if(index_new == len(feature_col)):
                break
            
            s_y = year_col.loc[index_new]
            s_m = month_col.loc[index_new]
            s_d = day_col.loc[index_new]
        
    return normalize(variety)
    
def mean_4_feature(feature_col, year_col, month_col, day_col):
    """
    mean of input feature of one day
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        var_temp = a.mean()
        
        # sleep(1)
        
        var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)
    
    
    
def max_4_feature(feature_col, year_col, month_col, day_col):
    """
    max of input feature of one day
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        var_temp = a.max()
        
        # sleep(1)
        
        var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)
    
def min_4_feature(feature_col, year_col, month_col, day_col):
    """
    min of input feature of one day
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        var_temp = a.min()
        
        # sleep(1)
        
        var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)
    
def var_4_feature(feature_col, year_col, month_col, day_col):
    """
    variety of input feature of one day
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        var_temp = a.max() - a.min()
        # var_temp = a.mean()
        
        # sleep(1)
        
        var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)

def daytime_min_4_feature(feature_col, re_irradiace_col, year_col, month_col, day_col):
    """
    variety of input feature of daytime
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        b = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (re_irradiace_col != -1)]
        
        var_temp = b.min()
        # var_temp = a.mean()
        
        # sleep(1)
        if len(a) == 1:
            var.loc[index_new : index_new + len(a) - 1, 'output'] = 0
        else:
            var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)
    
def daytime_max_4_feature(feature_col, re_irradiace_col, year_col, month_col, day_col):
    """
    variety of input feature of daytime
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        b = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (re_irradiace_col != -1)]
        
        var_temp = b.max()
        # var_temp = a.mean()
        
        # sleep(1)
        if len(a) == 1:
            var.loc[index_new : index_new + len(a) - 1, 'output'] = 0
        else:
            var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)
    
def daytime_mean_4_feature(feature_col, re_irradiace_col, year_col, month_col, day_col):
    """
    variety of input feature of daytime
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        b = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (re_irradiace_col != -1)]
        
        
        var_temp = b.mean()
        
        if len(a) == 1:
            var.loc[index_new : index_new + len(a) - 1, 'output'] = 0
        else:
            var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)
    
def daytime_var_4_feature(feature_col, re_irradiace_col, year_col, month_col, day_col):
    """
    variety of input feature of daytime
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        b = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (re_irradiace_col != -1)]
        
        # print(a)
        # print(b)
        
        
        var_temp = b.max() - b.min()
        # print(var_temp)
        # var_temp = a.mean()
        # os.system("pause")
        
        if len(a) == 1:
            var.loc[index_new : index_new + len(a) - 1, 'output'] = 0
        else:
            var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    # pd.set_option('display.max_rows', 600)
    # print(var)
    # os.system("pause")
    return normalize(var)
    
def nighttime_mean_4_feature(feature_col, irradiance_col, year_col, month_col, day_col):
    """
    mean of input feature of night
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        b = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (irradiance_col == -1)]
        
        var_temp = b.mean()
        
        # sleep(1)
        
        var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)
    
def nighttime_max_4_feature(feature_col, irradiance_col, year_col, month_col, day_col):
    """
    mean of input feature of night
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        b = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (irradiance_col == -1)]
        
        var_temp = b.max()
        
        # sleep(1)
        
        var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)
    
def nighttime_min_4_feature(feature_col, irradiance_col, year_col, month_col, day_col):
    """
    mean of input feature of night
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        b = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (irradiance_col == -1)]
        
        var_temp = b.min()
        
        # sleep(1)
        
        var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)
    
def nighttime_var_4_feature(feature_col, irradiance_col, year_col, month_col, day_col):
    """
    mean of input feature of night
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['output'] = np.zeros(len(feature_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(feature_col)):
        
        # index
        a = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d)]
        
        b = feature_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (irradiance_col == -1)]
        
        var_temp = b.max() - b.min()
        
        # sleep(1)
        
        var.loc[index_new : index_new + len(a) - 1, 'output'] = var_temp
        
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(feature_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    return normalize(var)
    
    
def daytime_feature(irradiance_col, year_col, month_col, day_col):
    """
    day time of one day
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['daytime'] = np.zeros(len(year_col))
    
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    while (index_new < len(year_col)):
        
        # index
        a = irradiance_col[(year_col == s_y) & (month_col == s_m) &(day_col == s_d)]
        
        d = irradiance_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (irradiance_col != -1)]
        
        
        if (len(d) == 1):
        
            var.loc[index_new : index_new + len(a) - 1, 'daytime'] = 0
            
        else:
            var.loc[index_new : index_new + len(a) - 1, 'daytime'] = len(d)
            
            
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(year_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
    
    return normalize(var)
    
def nighttime_feature(irradiance_col, year_col, month_col, day_col):
    """
    night time of one day
    editor : yyh
    """
    var = pd.DataFrame()
    
    
    var['nighttime'] = np.zeros(len(year_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    
    while (index_new < len(year_col)):
        
        # index
        a = irradiance_col[(year_col == s_y) & (month_col == s_m) &(day_col == s_d)]
        
        
        n = irradiance_col[(year_col == s_y) & (month_col == s_m) & (day_col == s_d) & (irradiance_col == -1)]
        
        if (len(n) == 1):
            
            var.loc[index_new : index_new + len(a) - 1, 'nighttime'] = len(n)
        else:
            
            var.loc[index_new : index_new + len(a) - 1, 'nighttime'] = len(n)
            
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(year_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
        
    
    return normalize(var)
    
def season_4_feature(year_col, month_col, day_col):
    """
    one hot of diffenrent season
    editor : yyh
    """
    var = pd.DataFrame()
    
    var['spring'] = np.zeros(len(year_col))
    
    var['summer'] = np.zeros(len(year_col))
    
    var['autumn'] = np.zeros(len(year_col))
    
    var['winter'] = np.zeros(len(year_col))
    
    index_new = 0
    
    s_y = year_col.loc[index_new]
    s_m = month_col.loc[index_new]
    s_d = day_col.loc[index_new]
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len(year_col)):
        
        # index
        a = year_col[(year_col == s_y) & (month_col == s_m) &(day_col == s_d)]
        
        
        if((s_m == 12) | (s_m == 1) | (s_m == 2)):
        
            var.loc[index_new : index_new + len(a) - 1, 'winter'] = 1
            var.loc[index_new : index_new + len(a) - 1, 'spring'] = 0
            var.loc[index_new : index_new + len(a) - 1, 'summer'] = 0
            var.loc[index_new : index_new + len(a) - 1, 'autumn'] = 0
        
        if((s_m == 3) | (s_m == 4) | (s_m == 5)):
        
            var.loc[index_new : index_new + len(a) - 1, 'winter'] = 0
            var.loc[index_new : index_new + len(a) - 1, 'spring'] = 1
            var.loc[index_new : index_new + len(a) - 1, 'summer'] = 0
            var.loc[index_new : index_new + len(a) - 1, 'autumn'] = 0
            
        if((s_m == 6) | (s_m == 7) | (s_m == 8)):
        
            var.loc[index_new : index_new + len(a) - 1, 'winter'] = 0
            var.loc[index_new : index_new + len(a) - 1, 'spring'] = 0
            var.loc[index_new : index_new + len(a) - 1, 'summer'] = 1
            var.loc[index_new : index_new + len(a) - 1, 'autumn'] = 0
            
        if((s_m == 9) | (s_m == 10) | (s_m == 11)):
        
            var.loc[index_new : index_new + len(a) - 1, 'winter'] = 0
            var.loc[index_new : index_new + len(a) - 1, 'spring'] = 0
            var.loc[index_new : index_new + len(a) - 1, 'summer'] = 0
            var.loc[index_new : index_new + len(a) - 1, 'autumn'] = 1
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len(year_col)):
            break
        
        s_y = year_col.loc[index_new]
        s_m = month_col.loc[index_new]
        s_d = day_col.loc[index_new]
    
    return normalize(var)
    
def data_missing_process_format(train_old):
    """
    deal with the missing data, delete or fill
    
    editor : yyh
    """
    
    print(train_old)
    os.system("pause")
    
    index_new = 0
    len_told = len(train_old)
    
    s_y = train_old['year'].loc[index_new]
    s_m = train_old['month'].loc[index_new]
    s_d = train_old['day'].loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len_told):
        
        # index
        a = train_old[(train_old['year'] == s_y) & (train_old['month'] == s_m) & (train_old['day'] == s_d)]
        
        
        light_max = a['实发辐照度'].max()
        
        if (a.loc[(a['实发辐照度'] == light_max), '实际功率'].values[0] < 1.5):
            
            print(a)
            print(light_max)
            print(a.loc[(a['实发辐照度'] == light_max), '实际功率'].values[0])
            print(a['实际功率'].max())
            
            train_old = train_old[~((train_old['year'].isin([s_y])) & (train_old['month'].isin([s_m])) & (train_old['day'].isin([s_d])))]
            # print(train_old)
            os.system("pause")
            
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len_told):
            break
        
        s_y = train_old['year'].loc[index_new]
        s_m = train_old['month'].loc[index_new]
        s_d = train_old['day'].loc[index_new]
        
    
    train_new = train_old.reset_index(drop = True)
    
    print(train_new)
    os.system("pause")
    
    return train_new
    

    
def data_missing_process1(train_old):
    """
    deal with the missing data, delete or fill
    #### del 2016-7-18 2017-10-21 2017-10-29 of train1
    editor : yyh
    """
    
    index_new = 0
    len_told = len(train_old)
    
    s_y = train_old['year'].loc[index_new]
    s_m = train_old['month'].loc[index_new]
    s_d = train_old['day'].loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len_told):
        
        # index
        a = train_old[(train_old['year'] == s_y) & (train_old['month'] == s_m) & (train_old['day'] == s_d)]
        
        
        light_max = a['实发辐照度'].max()
        
        if (a.loc[(a['实发辐照度'] == light_max), '实际功率'].values[0] < 1.5):
        
            train_old = train_old[~((train_old['year'].isin([s_y])) & (train_old['month'].isin([s_m])) & (train_old['day'].isin([s_d])))]
            
            
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len_told):
            break
        
        s_y = train_old['year'].loc[index_new]
        s_m = train_old['month'].loc[index_new]
        s_d = train_old['day'].loc[index_new]
        
    
    train_new = train_old.reset_index(drop = True)
    
    
    return train_new
    
def data_missing_process2(train_old):
    """
    deal with the missing data, delete or fill
    
    editor : yyh
    """
    
    
    index_new = 0
    len_told = len(train_old)
    
    s_y = train_old['year'].loc[index_new]
    s_m = train_old['month'].loc[index_new]
    s_d = train_old['day'].loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len_told):
        
        # index
        a = train_old[(train_old['year'] == s_y) & (train_old['month'] == s_m) & (train_old['day'] == s_d)]
        
        
        light_max = a['实发辐照度'].max()
        
        if ((a['实际功率'].max() < 1) | (a.loc[(a['实发辐照度'] == light_max), '实际功率'].values[0] < 0.2)):
        
            train_old = train_old[~((train_old['year'].isin([s_y])) & (train_old['month'].isin([s_m])) & (train_old['day'].isin([s_d])))]
            
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len_told):
            break
        
        s_y = train_old['year'].loc[index_new]
        s_m = train_old['month'].loc[index_new]
        s_d = train_old['day'].loc[index_new]
        
    
    train_new = train_old.reset_index(drop = True)
    
    return train_new
    
    
def data_missing_process3(train_old):
    """
    deal with the missing data, delete or fill
    
    editor : yyh
    """
    
    
    index_new = 0
    len_told = len(train_old)
    
    s_y = train_old['year'].loc[index_new]
    s_m = train_old['month'].loc[index_new]
    s_d = train_old['day'].loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len_told):
        
        # index
        a = train_old[(train_old['year'] == s_y) & (train_old['month'] == s_m) & (train_old['day'] == s_d)]
        
        
        light_max = a['实发辐照度'].max()
        
        if (a.loc[(a['实发辐照度'] == light_max), '实际功率'].values[0] < 0.2):
            
            
            train_old = train_old[~((train_old['year'].isin([s_y])) & (train_old['month'].isin([s_m])) & (train_old['day'].isin([s_d])))]
            # print(train_old)
            
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len_told):
            break
        
        s_y = train_old['year'].loc[index_new]
        s_m = train_old['month'].loc[index_new]
        s_d = train_old['day'].loc[index_new]
        
    
    train_new = train_old.reset_index(drop = True)
    
    return train_new
    
def data_missing_process4(train_old):
    """
    deal with the missing data, delete or fill
    
    editor : yyh
    """
    
    index_new = 0
    len_told = len(train_old)
    
    s_y = train_old['year'].loc[index_new]
    s_m = train_old['month'].loc[index_new]
    s_d = train_old['day'].loc[index_new]
    
    # pd.set_option('display.max_rows', 600)
    
    while (index_new < len_told):
        
        # index
        a = train_old[(train_old['year'] == s_y) & (train_old['month'] == s_m) & (train_old['day'] == s_d)]
        
        
        light_max = a['实发辐照度'].max()
        
        if (a.loc[(a['实发辐照度'] == light_max), '实际功率'].values[0] < 0.5):
            
            
            train_old = train_old[~((train_old['year'].isin([s_y])) & (train_old['month'].isin([s_m])) & (train_old['day'].isin([s_d])))]
            
        #update index
        index_new = index_new + len(a)
        
        if(index_new == len_told):
            break
        
        s_y = train_old['year'].loc[index_new]
        s_m = train_old['month'].loc[index_new]
        s_d = train_old['day'].loc[index_new]
        
    
    train_new = train_old.reset_index(drop = True)
    
    
    return train_new
    
def specialize_2(month_col, hour_col, min_col, label_col):
    """
    special condition which is set for the station 2
    set 0 according to different month
    editor : yyh
    """
    label_col[(month_col == 5) & (((hour_col == 20) & (min_col == 00)) | ((hour_col == 20) & (min_col == 15)) | ((hour_col == 20) & (min_col == 30)) | ((hour_col == 20) & (min_col == 45)) | ((hour_col == 21) & (min_col == 0)) | ((hour_col == 21) & (min_col == 15)) | ((hour_col == 21) & (min_col == 30)) | ((hour_col == 21) & (min_col == 45)) | ((hour_col == 22) & (min_col == 00)) | ((hour_col == 22) & (min_col == 15)) | ((hour_col == 22) & (min_col == 30)) | ((hour_col == 22) & (min_col == 45)) | ((hour_col == 23) & (min_col == 00)) | ((hour_col == 23) & (min_col == 15)) | ((hour_col == 23) & (min_col == 30)) | ((hour_col == 23) & (min_col == 45)) | ((hour_col == 0) & (min_col == 0)) | ((hour_col == 0) & (min_col == 15)) | ((hour_col == 0) & (min_col == 30)) | ((hour_col == 0) & (min_col == 45)) | ((hour_col == 1) & (min_col == 0)) | ((hour_col == 1) & (min_col == 15)) | ((hour_col == 1) & (min_col == 30)) | ((hour_col == 1) & (min_col == 45)) | ((hour_col == 2) & (min_col == 0)) | ((hour_col == 2) & (min_col == 15)) | ((hour_col == 2) & (min_col == 30)) | ((hour_col == 2) & (min_col == 45)) | ((hour_col == 3) & (min_col == 0)) | ((hour_col == 3) & (min_col == 15)) | ((hour_col == 3) & (min_col == 30)) | ((hour_col == 3) & (min_col == 45)) | ((hour_col == 4) & (min_col == 0)) | ((hour_col == 4) & (min_col == 15)) | ((hour_col == 4) & (min_col == 30)) | ((hour_col == 4) & (min_col == 45)) | ((hour_col == 5) & (min_col == 0)) | ((hour_col == 5) & (min_col == 15)) | ((hour_col == 5) & (min_col == 30)) | ((hour_col == 5) & (min_col == 45)))] = 0
    
    label_col[(month_col == 6) & (((hour_col == 20) & (min_col == 30)) | ((hour_col == 20) & (min_col == 45)) | ((hour_col == 21) & (min_col == 0)) | ((hour_col == 21) & (min_col == 15)) | ((hour_col == 21) & (min_col == 30)) | ((hour_col == 21) & (min_col == 45)) | ((hour_col == 22) & (min_col == 00)) | ((hour_col == 22) & (min_col == 15)) | ((hour_col == 22) & (min_col == 30)) | ((hour_col == 22) & (min_col == 45)) | ((hour_col == 23) & (min_col == 00)) | ((hour_col == 23) & (min_col == 15)) | ((hour_col == 23) & (min_col == 30)) | ((hour_col == 23) & (min_col == 45)) | ((hour_col == 0) & (min_col == 0)) | ((hour_col == 0) & (min_col == 15)) | ((hour_col == 0) & (min_col == 30)) | ((hour_col == 0) & (min_col == 45)) | ((hour_col == 1) & (min_col == 0)) | ((hour_col == 1) & (min_col == 15)) | ((hour_col == 1) & (min_col == 30)) | ((hour_col == 1) & (min_col == 45)) | ((hour_col == 2) & (min_col == 0)) | ((hour_col == 2) & (min_col == 15)) | ((hour_col == 2) & (min_col == 30)) | ((hour_col == 2) & (min_col == 45)) | ((hour_col == 3) & (min_col == 0)) | ((hour_col == 3) & (min_col == 15)) | ((hour_col == 3) & (min_col == 30)) | ((hour_col == 3) & (min_col == 45)) | ((hour_col == 4) & (min_col == 0)) | ((hour_col == 4) & (min_col == 15)) | ((hour_col == 4) & (min_col == 30)) | ((hour_col == 4) & (min_col == 45)) | ((hour_col == 5) & (min_col == 0)) | ((hour_col == 5) & (min_col == 15)) | ((hour_col == 5) & (min_col == 30)) | ((hour_col == 5) & (min_col == 45)))] = 0
    
    label_col[(month_col == 7) & (((hour_col == 20) & (min_col == 00)) | ((hour_col == 20) & (min_col == 15)) | ((hour_col == 20) & (min_col == 30)) | ((hour_col == 20) & (min_col == 45)) | ((hour_col == 21) & (min_col == 0)) | ((hour_col == 21) & (min_col == 15)) | ((hour_col == 21) & (min_col == 30)) | ((hour_col == 21) & (min_col == 45)) | ((hour_col == 22) & (min_col == 00)) | ((hour_col == 22) & (min_col == 15)) | ((hour_col == 22) & (min_col == 30)) | ((hour_col == 22) & (min_col == 45)) | ((hour_col == 23) & (min_col == 00)) | ((hour_col == 23) & (min_col == 15)) | ((hour_col == 23) & (min_col == 30)) | ((hour_col == 23) & (min_col == 45)) | ((hour_col == 0) & (min_col == 0)) | ((hour_col == 0) & (min_col == 15)) | ((hour_col == 0) & (min_col == 30)) | ((hour_col == 0) & (min_col == 45)) | ((hour_col == 1) & (min_col == 0)) | ((hour_col == 1) & (min_col == 15)) | ((hour_col == 1) & (min_col == 30)) | ((hour_col == 1) & (min_col == 45)) | ((hour_col == 2) & (min_col == 0)) | ((hour_col == 2) & (min_col == 15)) | ((hour_col == 2) & (min_col == 30)) | ((hour_col == 2) & (min_col == 45)) | ((hour_col == 3) & (min_col == 0)) | ((hour_col == 3) & (min_col == 15)) | ((hour_col == 3) & (min_col == 30)) | ((hour_col == 3) & (min_col == 45)) | ((hour_col == 4) & (min_col == 0)) | ((hour_col == 4) & (min_col == 15)) | ((hour_col == 4) & (min_col == 30)) | ((hour_col == 4) & (min_col == 45)) | ((hour_col == 5) & (min_col == 0)) | ((hour_col == 5) & (min_col == 15)) | ((hour_col == 5) & (min_col == 30)) | ((hour_col == 5) & (min_col == 45)) | ((hour_col == 6) & (min_col == 0)))] = 0
    
    label_col[(month_col == 8) & (((hour_col == 20) & (min_col == 15)) | ((hour_col == 20) & (min_col == 30)) | ((hour_col == 20) & (min_col == 45)) | ((hour_col == 21) & (min_col == 0)) | ((hour_col == 21) & (min_col == 15)) | ((hour_col == 21) & (min_col == 30)) | ((hour_col == 21) & (min_col == 45)) | ((hour_col == 22) & (min_col == 00)) | ((hour_col == 22) & (min_col == 15)) | ((hour_col == 22) & (min_col == 30)) | ((hour_col == 22) & (min_col == 45)) | ((hour_col == 23) & (min_col == 00)) | ((hour_col == 23) & (min_col == 15)) | ((hour_col == 23) & (min_col == 30)) | ((hour_col == 23) & (min_col == 45)) | ((hour_col == 0) & (min_col == 0)) | ((hour_col == 0) & (min_col == 15)) | ((hour_col == 0) & (min_col == 30)) | ((hour_col == 0) & (min_col == 45)) | ((hour_col == 1) & (min_col == 0)) | ((hour_col == 1) & (min_col == 15)) | ((hour_col == 1) & (min_col == 30)) | ((hour_col == 1) & (min_col == 45)) | ((hour_col == 2) & (min_col == 0)) | ((hour_col == 2) & (min_col == 15)) | ((hour_col == 2) & (min_col == 30)) | ((hour_col == 2) & (min_col == 45)) | ((hour_col == 3) & (min_col == 0)) | ((hour_col == 3) & (min_col == 15)) | ((hour_col == 3) & (min_col == 30)) | ((hour_col == 3) & (min_col == 45)) | ((hour_col == 4) & (min_col == 0)) | ((hour_col == 4) & (min_col == 15)) | ((hour_col == 4) & (min_col == 30)) | ((hour_col == 4) & (min_col == 45)) | ((hour_col == 5) & (min_col == 0)) | ((hour_col == 5) & (min_col == 15)) | ((hour_col == 5) & (min_col == 30)) | ((hour_col == 5) & (min_col == 45)) | ((hour_col == 6) & (min_col == 0)))] = 0
    
    label_col[(month_col == 9) & (((hour_col == 19) & (min_col == 45)) | ((hour_col == 20) & (min_col == 0)) | ((hour_col == 20) & (min_col == 15)) | ((hour_col == 20) & (min_col == 30)) | ((hour_col == 20) & (min_col == 45)) | ((hour_col == 21) & (min_col == 0)) | ((hour_col == 21) & (min_col == 15)) | ((hour_col == 21) & (min_col == 30)) | ((hour_col == 21) & (min_col == 45)) | ((hour_col == 22) & (min_col == 00)) | ((hour_col == 22) & (min_col == 15)) | ((hour_col == 22) & (min_col == 30)) | ((hour_col == 22) & (min_col == 45)) | ((hour_col == 23) & (min_col == 00)) | ((hour_col == 23) & (min_col == 15)) | ((hour_col == 23) & (min_col == 30)) | ((hour_col == 23) & (min_col == 45)) | ((hour_col == 0) & (min_col == 0)) | ((hour_col == 0) & (min_col == 15)) | ((hour_col == 0) & (min_col == 30)) | ((hour_col == 0) & (min_col == 45)) | ((hour_col == 1) & (min_col == 0)) | ((hour_col == 1) & (min_col == 15)) | ((hour_col == 1) & (min_col == 30)) | ((hour_col == 1) & (min_col == 45)) | ((hour_col == 2) & (min_col == 0)) | ((hour_col == 2) & (min_col == 15)) | ((hour_col == 2) & (min_col == 30)) | ((hour_col == 2) & (min_col == 45)) | ((hour_col == 3) & (min_col == 0)) | ((hour_col == 3) & (min_col == 15)) | ((hour_col == 3) & (min_col == 30)) | ((hour_col == 3) & (min_col == 45)) | ((hour_col == 4) & (min_col == 0)) | ((hour_col == 4) & (min_col == 15)) | ((hour_col == 4) & (min_col == 30)) | ((hour_col == 4) & (min_col == 45)) | ((hour_col == 5) & (min_col == 0)) | ((hour_col == 5) & (min_col == 15)) | ((hour_col == 5) & (min_col == 30)) | ((hour_col == 5) & (min_col == 45)) | ((hour_col == 6) & (min_col == 0)) | ((hour_col == 6) & (min_col == 15)) | ((hour_col == 6) & (min_col == 30)) | ((hour_col == 6) & (min_col == 45)) | ((hour_col == 7) & (min_col == 0)))] = 0
    
    return label_col
    
    
def specialize_3(month_col, hour_col, min_col, label_col):
    """
    special condition which is set for the station 3
    set 0 according to different month
    editor : yyh
    """
    label_col[(((hour_col == 20) & (min_col == 15)) | ((hour_col == 20) & (min_col == 30)) | ((hour_col == 20) & (min_col == 45)) | ((hour_col == 21) & (min_col == 0)) | ((hour_col == 21) & (min_col == 15)) | ((hour_col == 21) & (min_col == 30)) | ((hour_col == 21) & (min_col == 45)) | ((hour_col == 22) & (min_col == 00)) | ((hour_col == 22) & (min_col == 15)) | ((hour_col == 22) & (min_col == 30)) | ((hour_col == 22) & (min_col == 45)) | ((hour_col == 23) & (min_col == 00)) | ((hour_col == 23) & (min_col == 15)) | ((hour_col == 23) & (min_col == 30)) | ((hour_col == 23) & (min_col == 45)) | ((hour_col == 0) & (min_col == 0)) | ((hour_col == 0) & (min_col == 15)) | ((hour_col == 0) & (min_col == 30)) | ((hour_col == 0) & (min_col == 45)) | ((hour_col == 1) & (min_col == 0)) | ((hour_col == 1) & (min_col == 15)) | ((hour_col == 1) & (min_col == 30)) | ((hour_col == 1) & (min_col == 45)) | ((hour_col == 2) & (min_col == 0)) | ((hour_col == 2) & (min_col == 15)) | ((hour_col == 2) & (min_col == 30)) | ((hour_col == 2) & (min_col == 45)) | ((hour_col == 3) & (min_col == 0)) | ((hour_col == 3) & (min_col == 15)) | ((hour_col == 3) & (min_col == 30)) | ((hour_col == 3) & (min_col == 45)) | ((hour_col == 4) & (min_col == 0)) | ((hour_col == 4) & (min_col == 15)) | ((hour_col == 4) & (min_col == 30)) | ((hour_col == 4) & (min_col == 45)) | ((hour_col == 5) & (min_col == 0)) | ((hour_col == 5) & (min_col == 15)) | ((hour_col == 5) & (min_col == 30)) | ((hour_col == 5) & (min_col == 45)) | ((hour_col == 6) & (min_col == 0)) | ((hour_col == 6) & (min_col == 15)))] = 0
    
    
    return label_col
    
def specialize_4(month_col, hour_col, min_col, label_col):
    """
    special condition which is set for the station 4
    set 0 according to different month
    editor : yyh
    """
    
    label_col[(month_col == 5) & (((hour_col == 20) & (min_col == 30)) | ((hour_col == 20) & (min_col == 45)) | ((hour_col == 21) & (min_col == 0)) | ((hour_col == 21) & (min_col == 15)) | ((hour_col == 21) & (min_col == 30)) | ((hour_col == 21) & (min_col == 45)) | ((hour_col == 22) & (min_col == 00)) | ((hour_col == 22) & (min_col == 15)) | ((hour_col == 22) & (min_col == 30)) | ((hour_col == 22) & (min_col == 45)) | ((hour_col == 23) & (min_col == 00)) | ((hour_col == 23) & (min_col == 15)) | ((hour_col == 23) & (min_col == 30)) | ((hour_col == 23) & (min_col == 45)) | ((hour_col == 0) & (min_col == 0)) | ((hour_col == 0) & (min_col == 15)) | ((hour_col == 0) & (min_col == 30)) | ((hour_col == 0) & (min_col == 45)) | ((hour_col == 1) & (min_col == 0)) | ((hour_col == 1) & (min_col == 15)) | ((hour_col == 1) & (min_col == 30)) | ((hour_col == 1) & (min_col == 45)) | ((hour_col == 2) & (min_col == 0)) | ((hour_col == 2) & (min_col == 15)) | ((hour_col == 2) & (min_col == 30)) | ((hour_col == 2) & (min_col == 45)) | ((hour_col == 3) & (min_col == 0)) | ((hour_col == 3) & (min_col == 15)) | ((hour_col == 3) & (min_col == 30)) | ((hour_col == 3) & (min_col == 45)) | ((hour_col == 4) & (min_col == 0)) | ((hour_col == 4) & (min_col == 15)) | ((hour_col == 4) & (min_col == 30)) | ((hour_col == 4) & (min_col == 45)) | ((hour_col == 5) & (min_col == 0)) | ((hour_col == 5) & (min_col == 15)) | ((hour_col == 5) & (min_col == 30)) | ((hour_col == 5) & (min_col == 45)) | ((hour_col == 6) & (min_col == 0)) | ((hour_col == 6) & (min_col == 15)))] = 0
    
    label_col[(month_col == 6) & (((hour_col == 20) & (min_col == 45)) | ((hour_col == 21) & (min_col == 0)) | ((hour_col == 21) & (min_col == 15)) | ((hour_col == 21) & (min_col == 30)) | ((hour_col == 21) & (min_col == 45)) | ((hour_col == 22) & (min_col == 00)) | ((hour_col == 22) & (min_col == 15)) | ((hour_col == 22) & (min_col == 30)) | ((hour_col == 22) & (min_col == 45)) | ((hour_col == 23) & (min_col == 00)) | ((hour_col == 23) & (min_col == 15)) | ((hour_col == 23) & (min_col == 30)) | ((hour_col == 23) & (min_col == 45)) | ((hour_col == 0) & (min_col == 0)) | ((hour_col == 0) & (min_col == 15)) | ((hour_col == 0) & (min_col == 30)) | ((hour_col == 0) & (min_col == 45)) | ((hour_col == 1) & (min_col == 0)) | ((hour_col == 1) & (min_col == 15)) | ((hour_col == 1) & (min_col == 30)) | ((hour_col == 1) & (min_col == 45)) | ((hour_col == 2) & (min_col == 0)) | ((hour_col == 2) & (min_col == 15)) | ((hour_col == 2) & (min_col == 30)) | ((hour_col == 2) & (min_col == 45)) | ((hour_col == 3) & (min_col == 0)) | ((hour_col == 3) & (min_col == 15)) | ((hour_col == 3) & (min_col == 30)) | ((hour_col == 3) & (min_col == 45)) | ((hour_col == 4) & (min_col == 0)) | ((hour_col == 4) & (min_col == 15)) | ((hour_col == 4) & (min_col == 30)) | ((hour_col == 4) & (min_col == 45)) | ((hour_col == 5) & (min_col == 0)) | ((hour_col == 5) & (min_col == 15)) | ((hour_col == 5) & (min_col == 30)) | ((hour_col == 5) & (min_col == 45)) | ((hour_col == 6) & (min_col == 0)))] = 0
        
    label_col[(month_col == 7) & (((hour_col == 20) & (min_col == 45)) | ((hour_col == 21) & (min_col == 0)) | ((hour_col == 21) & (min_col == 15)) | ((hour_col == 21) & (min_col == 30)) | ((hour_col == 21) & (min_col == 45)) | ((hour_col == 22) & (min_col == 00)) | ((hour_col == 22) & (min_col == 15)) | ((hour_col == 22) & (min_col == 30)) | ((hour_col == 22) & (min_col == 45)) | ((hour_col == 23) & (min_col == 00)) | ((hour_col == 23) & (min_col == 15)) | ((hour_col == 23) & (min_col == 30)) | ((hour_col == 23) & (min_col == 45)) | ((hour_col == 0) & (min_col == 0)) | ((hour_col == 0) & (min_col == 15)) | ((hour_col == 0) & (min_col == 30)) | ((hour_col == 0) & (min_col == 45)) | ((hour_col == 1) & (min_col == 0)) | ((hour_col == 1) & (min_col == 15)) | ((hour_col == 1) & (min_col == 30)) | ((hour_col == 1) & (min_col == 45)) | ((hour_col == 2) & (min_col == 0)) | ((hour_col == 2) & (min_col == 15)) | ((hour_col == 2) & (min_col == 30)) | ((hour_col == 2) & (min_col == 45)) | ((hour_col == 3) & (min_col == 0)) | ((hour_col == 3) & (min_col == 15)) | ((hour_col == 3) & (min_col == 30)) | ((hour_col == 3) & (min_col == 45)) | ((hour_col == 4) & (min_col == 0)) | ((hour_col == 4) & (min_col == 15)) | ((hour_col == 4) & (min_col == 30)) | ((hour_col == 4) & (min_col == 45)) | ((hour_col == 5) & (min_col == 0)) | ((hour_col == 5) & (min_col == 15)) | ((hour_col == 5) & (min_col == 30)) | ((hour_col == 5) & (min_col == 45)) | ((hour_col == 6) & (min_col == 0)) | ((hour_col == 6) & (min_col == 15)))] = 0
    
    label_col[(month_col == 8) & (((hour_col == 20) & (min_col == 30)) | ((hour_col == 20) & (min_col == 45)) | ((hour_col == 21) & (min_col == 0)) | ((hour_col == 21) & (min_col == 15)) | ((hour_col == 21) & (min_col == 30)) | ((hour_col == 21) & (min_col == 45)) | ((hour_col == 22) & (min_col == 00)) | ((hour_col == 22) & (min_col == 15)) | ((hour_col == 22) & (min_col == 30)) | ((hour_col == 22) & (min_col == 45)) | ((hour_col == 23) & (min_col == 00)) | ((hour_col == 23) & (min_col == 15)) | ((hour_col == 23) & (min_col == 30)) | ((hour_col == 23) & (min_col == 45)) | ((hour_col == 0) & (min_col == 0)) | ((hour_col == 0) & (min_col == 15)) | ((hour_col == 0) & (min_col == 30)) | ((hour_col == 0) & (min_col == 45)) | ((hour_col == 1) & (min_col == 0)) | ((hour_col == 1) & (min_col == 15)) | ((hour_col == 1) & (min_col == 30)) | ((hour_col == 1) & (min_col == 45)) | ((hour_col == 2) & (min_col == 0)) | ((hour_col == 2) & (min_col == 15)) | ((hour_col == 2) & (min_col == 30)) | ((hour_col == 2) & (min_col == 45)) | ((hour_col == 3) & (min_col == 0)) | ((hour_col == 3) & (min_col == 15)) | ((hour_col == 3) & (min_col == 30)) | ((hour_col == 3) & (min_col == 45)) | ((hour_col == 4) & (min_col == 0)) | ((hour_col == 4) & (min_col == 15)) | ((hour_col == 4) & (min_col == 30)) | ((hour_col == 4) & (min_col == 45)) | ((hour_col == 5) & (min_col == 0)) | ((hour_col == 5) & (min_col == 15)) | ((hour_col == 5) & (min_col == 30)) | ((hour_col == 5) & (min_col == 45)) | ((hour_col == 6) & (min_col == 0)) | ((hour_col == 6) & (min_col == 15)))] = 0
    
    label_col[(month_col == 9) & (((hour_col == 20) & (min_col == 0)) | ((hour_col == 20) & (min_col == 15)) | ((hour_col == 20) & (min_col == 30)) | ((hour_col == 20) & (min_col == 45)) | ((hour_col == 21) & (min_col == 0)) | ((hour_col == 21) & (min_col == 15)) | ((hour_col == 21) & (min_col == 30)) | ((hour_col == 21) & (min_col == 45)) | ((hour_col == 22) & (min_col == 00)) | ((hour_col == 22) & (min_col == 15)) | ((hour_col == 22) & (min_col == 30)) | ((hour_col == 22) & (min_col == 45)) | ((hour_col == 23) & (min_col == 00)) | ((hour_col == 23) & (min_col == 15)) | ((hour_col == 23) & (min_col == 30)) | ((hour_col == 23) & (min_col == 45)) | ((hour_col == 0) & (min_col == 0)) | ((hour_col == 0) & (min_col == 15)) | ((hour_col == 0) & (min_col == 30)) | ((hour_col == 0) & (min_col == 45)) | ((hour_col == 1) & (min_col == 0)) | ((hour_col == 1) & (min_col == 15)) | ((hour_col == 1) & (min_col == 30)) | ((hour_col == 1) & (min_col == 45)) | ((hour_col == 2) & (min_col == 0)) | ((hour_col == 2) & (min_col == 15)) | ((hour_col == 2) & (min_col == 30)) | ((hour_col == 2) & (min_col == 45)) | ((hour_col == 3) & (min_col == 0)) | ((hour_col == 3) & (min_col == 15)) | ((hour_col == 3) & (min_col == 30)) | ((hour_col == 3) & (min_col == 45)) | ((hour_col == 4) & (min_col == 0)) | ((hour_col == 4) & (min_col == 15)) | ((hour_col == 4) & (min_col == 30)) | ((hour_col == 4) & (min_col == 45)) | ((hour_col == 5) & (min_col == 0)) | ((hour_col == 5) & (min_col == 15)) | ((hour_col == 5) & (min_col == 30)) | ((hour_col == 5) & (min_col == 45)) | ((hour_col == 6) & (min_col == 0)))] = 0
    
    return label_col
    
def _export_graphviz(train_data_x, label_y):
    """
    :return: 在我这边，在类中的函数，之前已经训练好了决策树self._clf_model,调用即可
    """
    clf = DecisionTreeRegressor(max_depth=3, max_leaf_nodes=10)
    clf.fit(train_data_x, label_y)
    dot_data = export_graphviz(clf, out_file=None)
    # print(clf.feature_importances_)
    graph = graphviz.Source(dot_data)
    graph.render("iris")

def final_train():
    model1 = LGBMRegressor(n_estimators=1600, objective='regression', learning_rate=0.01, random_state=50,
                           metric='rmse')
    model2 = XGBRegressor(n_estimators=16000, objective='reg:liner', learning_rate=0.01, n_jobs=-1, random_state=50,
                          eval_metric='rmse', silent=True)
    model1.fit(X_train, y_train)
    
    
    predict_label_1 = model1.predict(test)
    
    print(predict_label_1.shape)
    os.system("pause")
    df = pd.DataFrame(predict_label_1, columns = ['lgb_predict'])
    print(df)
    os.system("pause")
    df = pd.concat([test, df], axis = 1)
    pd.set_option('display.max_columns', None)
    print(df)
    os.system("pause")
    train_x['LGB_predict'] = predict_label_1
    test['LGB_predict']
    # print(train_x.info())
    model2.fit(train_x,train_y)
    predict_label_2 = model2.predict(test)
    final_predict = 0.5*predict_label_1+0.5*predict_label_2

    final_predict = pd.DataFrame(final_predict)
    sub = pd.concat([id, final_predict], axis=1)
    # print(sub.shape)
    sub.columns = ['id', 'predicition']
    sub.loc[sub['id'].isin(del_id), 'predicition'] = 0.0
    sub.to_csv(path + '/baseline4.csv', index=False, sep=',', encoding='UTF-8')

def shift_1(dt):
    """
    将dataframe的一列进行向下顺移一位
    同时将第一个数据用原始数据的第一个进行填充
    editor : yyh
    """
    dt_v = dt.values
    dt_v = dt_v.flatten()
    # print(dt_v)
    # print(dt_v.shape)
    # os.system("pause")
    i = len(dt)
    dt_new_v = np.zeros(i)
    dt_new_v[0] = dt_v[0]
    dt_new_v[1 : i] = dt_v[0 : i - 1]
    dt_new = pd.DataFrame()
    dt_new['test'] = dt_new_v
    # print(dt)
    # print('\n')
    # print(dt_new)
    # os.system("pause")
    return dt_new
    
def lgb_train_actual_irradiance(X_train, y_train, X_validation, y_validation, test_features, params, column_name, experiment_time, train_list):
    """
    column_name : a list who has onle one element which is a string
    editor : yyh
    """
    lgb_train = lgb.Dataset(X_train, label = y_train)
    lgb_eval = lgb.Dataset(X_validation, y_validation, reference = lgb_train)
    print('begin train')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=50000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100,
                    verbose_eval=100)
                    
    # plt.figure(figsize = (10, 15))
    
    lgb.plot_importance(gbm, max_num_features = 100, figsize = (8, 16))
    
    title = "FeatureImportance_actual_irradiance" +  '_' + train_list
    plt.title(title)
    
    save_path = 'J:/Code/DC/light/backup/feature_importance/' + experiment_time
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    name = title + '_' + train_list + '.jpg'
    
    save_path_name = save_path + '/' + name
    
    plt.savefig(save_path_name)
    
    pred_label = gbm.predict(test_features)
    
    
    pred_label = pd.DataFrame(pred_label, columns = column_name)

    return pred_label
    
def xgb_train_actual_irradiance(X_train, y_train, X_validation, y_validation, test_features, params, column_name, experiment_time, train_list):

    """
    column_name : a list who has onle one element which is a string
    editor : yyh
    """
    
    clf = XGBRegressor(max_depth = 6,
            learning_rate = 0.02,
            n_estimators = 160,
            silent = True,
            objective = 'reg:linear',
            booster = "gbtree",
            gamma = 0.1,
            min_child_weight = 1,
            subsample = 0.7,
            colsample_bytree = 0.5,
            reg_alpha = 0,
            reg_lamda = 10,
            random_state = 1000)
            
    print('begin train')
    
    clf.fit(X_train, y_train, eval_metric='auc')
    
    print("score : ", clf.score(X_validation, y_validation))
    
    pred_label = clf.predict(test_features)
    
    pred_label = pd.DataFrame(pred_label, columns = column_name)
    
    return pred_label
    
def lstm_train_actual_irradiance(X_train, y_train, X_validation, y_validation, test_features, params, column_name, experiment_time, train_list):

    """
    column_name : a list who has onle one element which is a string
    editor : yyh
    """
    X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
    
    X_validation = X_validation.values.reshape(X_validation.shape[0], 1, X_validation.shape[1])
    
    test_features = test_features.values.reshape(test_features.shape[0], 1, test_features.shape[1])
    
    model = Sequential()
    
    model.add(LSTM(96, activation = 'tanh', input_shape = (X_train.shape[1], X_train.shape[2])))
    
    model.add(Dense(1, activation = 'relu'))
    
    model.compile(loss = 'mse', optimizer = 'adam')
    
    model.fit(X_train, y_train, epochs = 100, batch_size = 100, validation_data = (X_validation, y_validation), verbose = 2, shuffle = False)
    
    y_validation_pred = model.predict(X_validation)
    
    pred_label = model.predict(test_features)
    
    rmse = mean_squared_error(y_validation, y_validation_pred) ** 0.5
    
    print("score : ", 1.0 / (1.0 + rmse))
    
    pred_label = model.predict(test_features)
    
    pred_label = pd.DataFrame(pred_label, columns = column_name)
    
    return pred_label
    
def NN_train_actual_irradiance(X_train, y_train, X_validation, y_validation, test_features, params, column_name, experiment_time, train_list):
    """
    column_name : a list who has onle one element which is a string
    editor : yyh
    """
    clf = MLPRegressor(hidden_layer_sizes=(100, 100, 100), activation= 'relu', solver='adam', alpha=0.0001, batch_size= 200, learning_rate='constant', learning_rate_init=0.02, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    clf.fit(X_train, y_train)
    
    y_validation_pred = clf.predict(X_validation)
    
    rmse = mean_squared_error(y_validation_pred, y_validation) ** 0.5
    
    print("score : ", 1.0 / (1.0 + rmse))
    
    pred_label = clf.predict(test_features)
    
    pred_label = pd.DataFrame(pred_label, columns = column_name)
    
    return pred_label
    
def lgb_train_actual_power(X_train, y_train, X_validation, y_validation, test_features, params, column_name, experiment_time, train_list):
    """
    column_name : a list who has onle one element which is a string
    editor : yyh
    """
    lgb_train = lgb.Dataset(X_train, label = y_train)
    lgb_eval = lgb.Dataset(X_validation, y_validation, reference = lgb_train)
    print('begin train')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=50000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100,
                    verbose_eval=100)
                    
    # plt.figure(figsize = (15, 10))
    
    lgb.plot_importance(gbm, max_num_features = 100, figsize = (8, 16))
    
    title = "FeatureImportance_actual_power" + '_' + train_list
    plt.title(title)
    
    save_path = 'J:/Code/DC/light/backup/feature_importance/' + experiment_time
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    name = title + '_' + train_list + '.jpg'
    
    save_path_name = save_path + '/' + name
    
    plt.savefig(save_path_name)
    
    pred_label = gbm.predict(test_features)
    
    
    pred_label = pd.DataFrame(pred_label, columns = column_name)

    return pred_label
    
def xgb_train_actual_power(X_train, y_train, X_validation, y_validation, test_features, params, column_name, experiment_time, train_list):

    """
    column_name : a list who has onle one element which is a string
    editor : yyh
    """
    clf = XGBRegressor(max_depth = 6,
            learning_rate = 0.02,
            n_estimators = 160,
            silent = True,
            objective = 'reg:linear',
            booster = "gbtree",
            gamma = 0.1,
            min_child_weight = 1,
            subsample = 0.7,
            colsample_bytree = 0.5,
            reg_alpha = 0,
            reg_lamda = 10,
            random_state = 1000)
            
    print('begin train')
    
    clf.fit(X_train, y_train, eval_metric='auc')
    
    print("score : ", clf.score(X_validation, y_validation))
    
    pred_label = clf.predict(test_features)
    
    pred_label = pd.DataFrame(pred_label, columns = column_name)
    
    return pred_label
    
def lstm_train_actual_power(X_train, y_train, X_validation, y_validation, test_features, params, column_name, experiment_time, train_list):

    """
    column_name : a list who has onle one element which is a string
    editor : yyh
    """
    X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
    
    X_validation = X_validation.values.reshape(X_validation.shape[0], 1, X_validation.shape[1])
    
    test_features = test_features.values.reshape(test_features.shape[0], 1, test_features.shape[1])
    
    model = Sequential()
    
    model.add(LSTM(96, activation = 'tanh', input_shape = (X_train.shape[1], X_train.shape[2])))
    
    model.add(Dense(1, activation = 'relu'))
    
    model.compile(loss = 'mse', optimizer = 'adam')
    
    model.fit(X_train, y_train, epochs = 100, batch_size = 100, validation_data = (X_validation, y_validation), verbose = 2, shuffle = False)
    
    y_validation_pred = model.predict(X_validation)
    
    pred_label = model.predict(test_features)
    
    rmse = mean_squared_error(y_validation, y_validation_pred) ** 0.5
    
    print("score : ", 1.0 / (1.0 + rmse))
    
    pred_label = model.predict(test_features)
    
    pred_label = pd.DataFrame(pred_label, columns = column_name)
    
    return pred_label
    
def NN_train_actual_power(X_train, y_train, X_validation, y_validation, test_features, params, column_name, experiment_time, train_list):
    """
    column_name : a list who has onle one element which is a string
    editor : yyh
    """
    clf = MLPRegressor(hidden_layer_sizes=(100, 100, 100), activation= 'relu', solver='adam', alpha=0.0001, batch_size= 200, learning_rate='constant', learning_rate_init=0.02, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    clf.fit(X_train, y_train)
    
    y_validation_pred = clf.predict(X_validation)
    
    rmse = mean_squared_error(y_validation_pred, y_validation) ** 0.5
    
    print("score : ", 1.0 / (1.0 + rmse))
    
    pred_label = clf.predict(test_features)
    
    pred_label = pd.DataFrame(pred_label, columns = column_name)
    
    return pred_label