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

def first_feature_engineering(train_old):

    ##########################################################
    ################ load train & data preprocessing##########
    ##########################################################
    
    train_features = pd.DataFrame()
    
    
    train_old['year'] = train_old['时间'].apply(lambda x: x[0 : 4]).astype('int32')
    
    train_old['month'] = train_old['时间'].apply(lambda x: get_month(x, train_list[i])).astype('int32')
    
    train_old['day'] = train_old['时间'].apply(lambda x: get_day(x, train_list[i])).astype('int32')

    
    train_old['hour'] = train_old['时间'].apply(lambda x: get_hour(x)).astype('int32')
    
    train_old['min'] = train_old['时间'].apply(lambda x: get_min(x)).astype('int32')
    
    train_features['month'] = normalize(train_old['month'])
    

    train_features['day'] = normalize(train_old['day'])

    train_features['time'] = normalize(train_old['时间'].apply(lambda x: get_time(x)).astype('int32'))
    
    
    train_features['辐照度'] = normalize(train_old['辐照度'])
    
    train_features['风速'] = normalize(train_old['风速'])
    
    
    train_features['风向'] =  normalize(train_old['风向'])

    train_features['温度'] =  normalize(train_old['温度'])

    train_features['压强'] =  normalize(train_old['压强'])

    train_features['湿度'] =  normalize(train_old['湿度'])
    
    
    train_features['dis2peak_辐照度'] = normalize(dis2peak(train_old['时间'].apply(lambda x: get_time(x)).astype('int32'), train_old['辐照度'], train_old['year'], train_old['month'], train_old['day']))

    train_features = add_poly_features(train_features, ['辐照度', '风速', '风向', '温度', '压强', '湿度']) ####
    
    train_features = add_plus_features(train_features, ['辐照度', '风速', '风向', '温度', '压强', '湿度'])
    
    train_features = add_div_features(train_features, ['辐照度', '风速', '风向', '温度', '压强', '湿度'])
    
    label_final = train_old['实际功率'] # label
    
    label_1 = normalize(train_old['实发辐照度'] )
    
    ###### new features
    ###temperature
    train_features['温度差'] = numerical_4_feature(train_old['温度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'allday')
    
    train_features['白天温度差'] = numerical_4_feature(train_old['温度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'daytime')
    
    train_features['夜晚温度差'] = numerical_4_feature(train_old['温度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'nighttime')
    
    train_features['温度std'] = numerical_4_feature(train_old['温度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'allday')
    
    train_features['白天温度std'] = numerical_4_feature(train_old['温度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'daytime')
    
    train_features['夜晚温度std'] = numerical_4_feature(train_old['温度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'nighttime')
    
    train_features['温度mean'] = numerical_4_feature(train_old['温度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'allday')
    
    train_features['白天温度mean'] = numerical_4_feature(train_old['温度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    train_features['夜晚温度mean'] = numerical_4_feature(train_old['温度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'nighttime')
    
    train_features = add_poly_features(train_features, ['温度差', '白天温度差', '夜晚温度差', '温度std', '白天温度std', '夜晚温度std', '温度mean', '白天温度mean', '夜晚温度mean'])
    
    ### humidity
    train_features['湿度差'] = numerical_4_feature(train_old['湿度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'allday')
    
    train_features['白天湿度差'] = numerical_4_feature(train_old['湿度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'daytime')
    
    train_features['夜晚湿度差'] = numerical_4_feature(train_old['湿度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'nighttime')
    
    train_features['湿度std'] = numerical_4_feature(train_old['湿度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'allday')
    
    train_features['白天湿度std'] = numerical_4_feature(train_old['湿度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'daytime')
    
    train_features['夜晚湿度std'] = numerical_4_feature(train_old['湿度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'nighttime')
    
    train_features['湿度mean'] = numerical_4_feature(train_old['温度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'allday')
    
    train_features['白天湿度mean'] = numerical_4_feature(train_old['湿度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    train_features['夜晚湿度mean'] = numerical_4_feature(train_old['湿度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'nighttime')
    
    ### pressure
    train_features['压强差'] = numerical_4_feature(train_old['压强'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'allday')
    
    train_features['白天压强差'] = numerical_4_feature(train_old['压强'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'daytime')
    
    train_features['夜晚压强差'] = numerical_4_feature(train_old['压强'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'nighttime')
    
    train_features['压强std'] = numerical_4_feature(train_old['压强'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'allday')
    
    train_features['白天压强std'] = numerical_4_feature(train_old['压强'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'daytime')
    
    train_features['夜晚压强std'] = numerical_4_feature(train_old['压强'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'nighttime')
    
    train_features['压强mean'] = numerical_4_feature(train_old['压强'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'allday')
    
    train_features['白天压强mean'] = numerical_4_feature(train_old['压强'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    train_features['夜晚压强mean'] = numerical_4_feature(train_old['压强'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'nighttime')
    
    ### wind speed
    train_features['风速差'] = numerical_4_feature(train_old['风速'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'allday')
    
    train_features['白天风速差'] = numerical_4_feature(train_old['风速'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'daytime')
    
    train_features['夜晚风速差'] = numerical_4_feature(train_old['风速'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'nighttime')
    
    train_features['风速std'] = numerical_4_feature(train_old['风速'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'allday')
    
    train_features['白天风速std'] = numerical_4_feature(train_old['风速'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'daytime')
    
    train_features['夜晚风速std'] = numerical_4_feature(train_old['风速'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'nighttime')
    
    train_features['风速mean'] = numerical_4_feature(train_old['风速'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'allday')
    
    train_features['白天风速mean'] = numerical_4_feature(train_old['风速'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    train_features['夜晚风速mean'] = numerical_4_feature(train_old['风速'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'nighttime')
    
    ###irradiance
    train_features['max辐照度'] = numerical_4_feature(train_old['辐照度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'max', time_period = 'allday')
    
    train_features['白天辐照度差'] =  numerical_4_feature(train_old['辐照度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'var', time_period = 'daytime')
    
    train_features['白天辐照度mean'] =  numerical_4_feature(train_old['辐照度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    train_features['白天辐照度std'] =  numerical_4_feature(train_old['辐照度'], train_old['year'], train_old['month'], train_old['day'], train_old['辐照度'], method = 'std', time_period = 'daytime')
    
    print(train_list[i])
    print(train_features.columns.tolist())
    print(len(train_features.columns.tolist()))
    
    #########################################################
    #############load test & data preprocessing##############
    #########################################################
    
    test_old = pd.read_csv(path + test_list[i])
    test_features = pd.DataFrame()
    
    test_old['year'] = test_old['时间'].apply(lambda x: x[0 : 4]).astype('int32')
    
    test_old['month'] = test_old['时间'].apply(lambda x: get_month(x, test_list[i])).astype('int32')
    
    test_old['day'] = test_old['时间'].apply(lambda x: get_day(x, test_list[i])).astype('int32')
    
    test_old['hour'] = test_old['时间'].apply(lambda x: get_hour(x)).astype('int32')
    
    test_old['min'] = test_old['时间'].apply(lambda x: get_min(x)).astype('int32')
    
    test_features['month'] = normalize(test_old['时间'].apply(lambda x: x[5:7]).astype('int32'))

    test_features['day'] = normalize(test_old['时间'].apply(lambda x: x[8:10]).astype('int32'))

    test_features['time'] = normalize(test_old['时间'].apply(lambda x: get_time(x)).astype('int32'))
    
    test_features['dis2peak_辐照度'] = normalize(dis2peak(test_old['时间'].apply(lambda x: get_time(x)).astype('int32'), test_old['辐照度'], test_old['year'], test_old['month'], test_old['day']))
    
    test_features['辐照度'] = normalize(test_old['辐照度'])
    
    test_features['风速'] =  normalize(test_old['风速'])

    test_features['风向'] =  normalize(test_old['风向'])

    test_features['温度'] =  normalize(test_old['温度'])

    test_features['压强'] =  normalize(test_old['压强'])

    test_features['湿度'] =  normalize(test_old['湿度'])

    test_features = add_poly_features(test_features, ['辐照度', '风速', '风向', '温度', '压强', '湿度'])
    
    
    
    test_features = add_plus_features(test_features, ['辐照度', '风速', '风向', '温度', '压强', '湿度'])
    
    
    test_features = add_div_features(test_features, ['辐照度', '风速', '风向', '温度', '压强', '湿度'])
    
    
    ##################### new features
    ###temperature
    test_features['温度差'] = numerical_4_feature(test_old['温度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'allday')
    
    test_features['白天温度差'] = numerical_4_feature(test_old['温度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'daytime')
    
    test_features['夜晚温度差'] = numerical_4_feature(test_old['温度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'nighttime')
    
    test_features['温度std'] = numerical_4_feature(test_old['温度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'daytime')
    
    test_features['白天温度std'] = numerical_4_feature(test_old['温度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'daytime')
    
    test_features['夜晚温度std'] = numerical_4_feature(test_old['温度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'nighttime')
    
    test_features['温度mean'] = numerical_4_feature(test_old['温度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'allday')
    
    test_features['白天温度mean'] = numerical_4_feature(test_old['温度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    test_features['夜晚温度mean'] = numerical_4_feature(test_old['温度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'nighttime')
    
    test_features = add_poly_features(test_features, ['温度差', '白天温度差', '夜晚温度差', '温度std', '白天温度std', '夜晚温度std', '温度mean', '白天温度mean', '夜晚温度mean'])
    
    ###humidity
    test_features['湿度差'] = numerical_4_feature(test_old['湿度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'allday')
    
    test_features['白天湿度差'] = numerical_4_feature(test_old['湿度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'daytime')
    
    test_features['夜晚湿度差'] = numerical_4_feature(test_old['湿度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'nighttime')
    
    test_features['湿度std'] = numerical_4_feature(test_old['湿度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'allday')
    
    test_features['白天湿度std'] = numerical_4_feature(test_old['湿度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'daytime')
    
    test_features['夜晚湿度std'] = numerical_4_feature(test_old['湿度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'nighttime')
    
    test_features['湿度mean'] = numerical_4_feature(test_old['温度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'allday')
    
    test_features['白天湿度mean'] = numerical_4_feature(test_old['湿度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    test_features['夜晚湿度mean'] = numerical_4_feature(test_old['湿度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'nighttime')
    
    
    ### pressure
    test_features['压强差'] = numerical_4_feature(test_old['压强'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'allday')
    
    test_features['白天压强差'] = numerical_4_feature(test_old['压强'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'daytime')
    
    test_features['夜晚压强差'] = numerical_4_feature(test_old['压强'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'nighttime')
    
    test_features['压强std'] = numerical_4_feature(test_old['压强'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'allday')
    
    test_features['白天压强std'] = numerical_4_feature(test_old['压强'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'daytime')
    
    test_features['夜晚压强std'] = numerical_4_feature(test_old['压强'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'nighttime')
    
    test_features['压强mean'] = numerical_4_feature(test_old['压强'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'allday')
    
    test_features['白天压强mean'] = numerical_4_feature(test_old['压强'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    test_features['夜晚压强mean'] = numerical_4_feature(test_old['压强'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'nighttime')
    
    
    ### wind speed 
    test_features['风速差'] = numerical_4_feature(test_old['风速'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'allday')
    
    test_features['白天风速差'] = numerical_4_feature(test_old['风速'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'daytime')
    
    test_features['夜晚风速差'] = numerical_4_feature(test_old['风速'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'nighttime')
    
    test_features['风速std'] = numerical_4_feature(test_old['风速'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'allday')
    
    test_features['白天风速std'] = numerical_4_feature(test_old['风速'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'daytime')
    
    test_features['夜晚风速std'] = numerical_4_feature(test_old['风速'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'nighttime')
    
    test_features['风速mean'] = numerical_4_feature(test_old['风速'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'allday')
    
    test_features['白天风速mean'] = numerical_4_feature(test_old['风速'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    test_features['夜晚风速mean'] = numerical_4_feature(test_old['风速'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'nighttime')
    
    ### irradiance
    test_features['max辐照度'] = numerical_4_feature(test_old['辐照度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'max', time_period = 'allday')
    
    test_features['白天辐照度差'] =  numerical_4_feature(test_old['辐照度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'var', time_period = 'daytime')
    
    test_features['白天辐照度mean'] =  numerical_4_feature(test_old['辐照度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'mean', time_period = 'daytime')
    
    test_features['白天辐照度std'] =  numerical_4_feature(test_old['辐照度'], test_old['year'], test_old['month'], test_old['day'], test_old['辐照度'], method = 'std', time_period = 'daytime')
    
    
    print(test_list[i])
    print(test_features.columns.tolist())
    print(len(test_features.columns.tolist()))
    
    if (len(train_features.columns) != len(test_features.columns)):
    
        print("\n \n warning : \n please check your features in your first train!!! \n train features and test features don't match to each other!!! \n")
        
        print("train features : ", train_features.columns.tolist())
        print("test features : ", test_features.columns.tolist())
        os.system("pause")
    
    return train_features, test_features, label_1, label_final