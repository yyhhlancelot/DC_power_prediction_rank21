#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
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

def second_feature_engineering(train_features, label_1, test_features, pred_label_1):
    ###########new train & test features
    ### train new
    train_features_new = pd.concat([train_features, label_1], axis = 1)
    
    train_features_new['max实发辐照度'] = numerical_4_feature(train_features_new['实发辐照度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'max', time_period = 'allday')
    
    train_features_new['白天实发辐照度mean'] = numerical_4_feature(train_features_new['实发辐照度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    train_features_new['白天实发辐照度std'] = numerical_4_feature(train_features_new['实发辐照度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'daytime')
    
    train_features_new['白天实发辐照度差'] = numerical_4_feature(train_features_new['实发辐照度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'daytime')
    
    train_features_new['dis2peak_实发辐照度'] = dis2peak(train_old['时间'].apply(lambda x: get_time(x)).astype('int32'), train_old['实发辐照度'], train_old['year'], train_old['month'], train_old['day'])
    
    
    train_features_new['平均辐照度'] = numerical_4_feature(train_features_new['辐照度'], train_old['year'], train_old['month'], train_old['day'], train_old['实发辐照度'], method = 'mean', time_period = 'allday')
    
    print(train_features_new.columns.tolist())
    print(len(train_features_new.columns.tolist()))
    
    ### test new
    test_features_new = pd.concat([test_features, pred_label_1], axis = 1)
    
    test_features_new['max实发辐照度'] = numerical_4_feature(test_features_new['实发辐照度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'max', time_period = 'allday')
    
    test_features_new['白天实发辐照度mean'] = numerical_4_feature(test_features_new['实发辐照度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    test_features_new['白天实发辐照度std'] = numerical_4_feature(test_features_new['实发辐照度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'daytime')
    
    test_features_new['白天实发辐照度差'] = numerical_4_feature(test_features_new['实发辐照度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'daytime')
    
    test_features_new['dis2peak_实发辐照度'] = dis2peak(test_old['时间'].apply(lambda x: get_time(x)).astype('int32'), test_features_new['实发辐照度'], test_old['year'], test_old['month'], test_old['day'])
    
    
    test_features_new['平均辐照度'] = numerical_4_feature(test_features_new['辐照度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'allday')
    
    if (len(train_features_new.columns) != len(test_features_new.columns)):
        
        print("warning : please check your features in your second train!!! \n train features and test features don't match to each other!!!")
        
        print(train_features_new.columns.tolist())
        print(test_features_new.columns.tolist())
        os.system("pause")
    
    return train_features_new, test_features_new