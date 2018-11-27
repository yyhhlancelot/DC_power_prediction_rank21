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
from func import get_hour, get_min, get_time, get_month, get_day, add_poly_features, add_plus_features, add_sub_features, add_div_features, dis2peak, lgb_train_actual_irradiance, lgb_train_actual_power, xgb_train_actual_irradiance, xgb_train_actual_power, normalize, shift_1, t_sne, pca, var_4_feature, mean_4_feature, max_4_feature, min_4_feature, daytime_var_4_feature, daytime_mean_4_feature, daytime_max_4_feature, daytime_min_4_feature, numerical_4_feature, daytime_feature, nighttime_feature, season_4_feature, specialize_2,specialize_3, specialize_4, data_missing_process1, data_missing_process2, data_missing_process3, data_missing_process4, data_selection, first_feature_engineering, second_feature_engineering, data_preprocessing
import time

def specialize(train_list_i, pred_label_final):
    ###specialize_2
    if(train_list[i] == 'train_2.csv'):
        
        pred_label_final = specialize_2(test_old['month'], test_old['hour'], test_old['min'], pred_label_final)
        
    ###specialize_3
    if(train_list[i] == 'train_3.csv'):
        
        pred_label_final = specialize_3(test_old['month'],  test_old['hour'], test_old['min'], pred_label_final)
        
    ###specialize_4
    if(train_list[i] == 'train_4.csv'):
        
        pred_label_final = specialize_4(test_old['month'], test_old['hour'], test_old['min'], pred_label_final)
        
    return pred_label_final