import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.tree import export_graphviz
import graphviz
import sys
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from func import get_hour, get_min, get_time, get_month, get_day, add_poly_features, add_plus_features, add_sub_features, add_div_features, dis2peak, lgb_train_actual_irradiance, lgb_train_actual_power, xgb_train_actual_irradiance, xgb_train_actual_power, normalize, shift_1, t_sne, pca, var_4_feature, mean_4_feature, max_4_feature, min_4_feature, daytime_var_4_feature, daytime_mean_4_feature, daytime_max_4_feature, daytime_min_4_feature, numerical_4_feature, daytime_feature, nighttime_feature, season_4_feature, specialize_2,specialize_3, specialize_4, data_missing_process1, data_missing_process2, data_missing_process3, data_missing_process4, data_selection
import time

def data_preprocessing(path, train_list_i, test_list_i):

    train_old = pd.read_csv(path + train_list_i)
    
    test_old = pd.read_csv(path + test_list_i)
    
    train_old['year'] = train_old['时间'].apply(lambda x: x[0 : 4]).astype('int32')
    
    train_old['month'] = train_old['时间'].apply(lambda x: get_month(x, train_list[i])).astype('int32')
    
    train_old['day'] = train_old['时间'].apply(lambda x: get_day(x, train_list[i])).astype('int32')
    
    #################data preprocessing###############
    
    #datamissing and fault 1
    if (train_list_i == 'train_1.csv'):
        
        train_old = train_old.drop([0])
        train_old = train_old.reset_index(drop = True)
        
        train_old = train_old[~(train_old['year'].isin([2018]) & train_old['month'].isin([4]) & train_old['day'].isin([1]))]
        
        train_old = train_old.reset_index(drop = True)
        
        train_old = data_missing_process1(train_old)
        
    #datamissing and fault 2
    if (train_list_i == 'train_2.csv'):
        
        train_old = train_old[~(train_old['month'].isin([4]) & (train_old['day'].isin([10]) | train_old['day'].isin([11]) | train_old['day'].isin([12])))]
        
        train_old = train_old.reset_index(drop = True)
        
        train_old = data_missing_process2(train_old)
        
    #datamissing and fault 3
    if (train_list_i == 'train_3.csv'):
        
        train_old = train_old[~(train_old['month'].isin([8]) & (train_old['day'].isin([4]) | train_old['day'].isin([5]) | train_old['day'].isin([6])))]
        
        train_old = train_old.reset_index(drop = True)
        
        train_old = data_missing_process3(train_old)
        
        
    #datamissing and fault 4
    if (train_list_i == 'train_4.csv'):
        
        train_old = data_missing_process4(train_old)
        
    return train_old, test_old