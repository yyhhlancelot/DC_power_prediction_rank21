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
from func import get_hour, get_min, get_time, get_month, get_day, add_poly_features, add_plus_features, add_sub_features, add_div_features, dis2peak, lgb_train_actual_irradiance, lgb_train_actual_power, xgb_train_actual_irradiance, xgb_train_actual_power, normalize, shift_1, t_sne, pca, var_4_feature, mean_4_feature, max_4_feature, min_4_feature, daytime_var_4_feature, daytime_mean_4_feature, daytime_max_4_feature, daytime_min_4_feature, numerical_4_feature, daytime_feature, nighttime_feature, season_4_feature, specialize_2,specialize_3, specialize_4, data_missing_process1, data_missing_process2, data_missing_process3, data_missing_process4, data_selection, first_feature_engineering, second_feature_engineering, data_preprocessing, specialize
import time

if __name__ == '__main__':
    
    start_time = time.time()
    
    ### 常量设置
    path = 'J:/Code/DC/light/backup/'
    train_list = ['train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv']
    test_list = ['test_1.csv', 'test_2.csv', 'test_3.csv', 'test_4.csv']
    index_list = ['1', '2', '3', '4']
    experiment_time = '21'
    
    lgb_params = {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 50,
        "min_child_samples": 100,
        "learning_rate": 0.02,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.5,
        "bagging_frequency": 5,
        "bagging_seed": 666,
        "verbosity": -1
    }
    
    xgb_params = {} #这里的参数在函数内部设置好了
    
    ### 对每个电场的数据依次进行训练
    for i in range(len(train_list)):
        
        ### 数据预处理
        train_old, test_old = data_preprocessing(path, train_list[i], test_list[i])
        
        ### 第一次训练的特征工程
        train_features, test_features, label_1, label_final= first_feature_engineering(train_old, test_old)
        
        ### 第一次训练 训练集和验证集划分
        X_train_1, X_validation_1, y_train_1, y_validation_1 = train_test_split(train_features, label_1, test_size=0.1, random_state=678)

        ### lgb第一次训练
        pred_label_1_lgb = lgb_train_actual_irradiance(X_train_1, y_train_1, X_validation_1, y_validation_1, test_features, lgb_params, ['实发辐照度'], experiment_time, train_list[i])
        
        ### xgb第一次训练
        pred_label_1_xgb = xgb_train_actual_irradiance(X_train_1, y_train_1, X_validation_1, y_validation_1, test_features, xgb_params, ['实发辐照度'], experiment_time, train_list[i])
        
        ### 对lgb训练得到的结果（实发辐照度）作为新特征加入测试集特征
        train_features_new, test_features_new_lgb = second_feature_engineering(train_features, label_1, test_features, pred_label_1_lgb)
        
        ### 对xgb训练得到的结果（实发辐照度）作为新特征加入测试集特征
        train_features_new, test_features_new_xgb = second_feature_engineering(train_features, label_1, test_features, pred_label_1_xgb)
        
        ### 第二次训练 训练集和验证集划分
        X_train_final, X_validation_final, y_train_final, y_validation_final = train_test_split(train_features_new, label_final, test_size=0.1, random_state=678)
        
        ### 得到lgb训练的预测实际功率
        pred_label_final_lgb = lgb_train_actual_power(X_train_final, y_train_final, X_validation_final, y_validation_final, test_features_new_lgb, lgb_params, ['predicition'], experiment_time, train_list[i])
        
        ### 得到xgb训练的预测实际功率
        pred_label_final_xgb = xgb_train_actual_power(X_train_final, y_train_final, X_validation_final, y_validation_final, test_features_new_xgb, xgb_params, ['predicition'], experiment_time, train_list[i])
        
        ### 对结果进行特殊化处理 进一步减小误差
        pred_label_final_lgb = specialize(train_list_i, pred_label_final_lgb)
        
        ### 对结果进行特殊化处理 进一步减小误差
        pred_label_final_xgb = specialize(train_list_i, pred_label_final_xgb)
        
        ### 存储lgb的每个训练集的结果
        result_single_name_lgb = '/result_lgb_' + index_list[i] + '_' + experiment_time + '.csv'
        
        ### 存储xgb的每个训练集的结果
        result_single_name_xgb = '/result_xgb_' + index_list[i] + '_' + experiment_time + '.csv'
        
        ### 结果标准化处理
        id = test_old['id']
        
        result_single_lgb = pd.concat([id, pred_label_final_lgb], axis = 1)
        
        result_single_xgb = pd.concat([id, pred_label_final_xgb], axis = 1)
        
        result_single_lgb.to_csv(path + result_single_name_lgb, index = False, sep = ',', encoding = 'UTF-8')
        
        result_single_xgb.to_csv(path + result_single_name_xgb, index = False, sep = ',', encoding = 'UTF-8')
    
    ### 结果整合
    
    result_1_lgb = pd.read_csv(path + '/result_lgb_1_' + experiment_time +'.csv')
    result_2_lgb = pd.read_csv(path + '/result_lgb_2_' + experiment_time +'.csv')
    result_3_lgb = pd.read_csv(path + '/result_lgb_3_' + experiment_time +'.csv')
    result_4_lgb = pd.read_csv(path + '/result_lgb_4_' + experiment_time +'.csv')
    
    result_lgb = pd.concat([result_1_lgb, result_2_lgb, result_3_lgb, result_4_lgb], axis = 0)
    result_lgb.to_csv(path + '/result_total_lgb_' + experiment_time + '.csv', index = False, sep = ',', encoding = 'UTF-8')
    
    result_1_xgb = pd.read_csv(path + '/result_xgb_1_' + experiment_time +'.csv')
    result_2_xgb = pd.read_csv(path + '/result_xgb_2_' + experiment_time +'.csv')
    result_3_xgb = pd.read_csv(path + '/result_xgb_3_' + experiment_time +'.csv')
    result_4_xgb = pd.read_csv(path + '/result_xgb_4_' + experiment_time +'.csv')
    
    result_xgb = pd.concat([result_1_xgb, result_2_xgb, result_3_xgb, result_4_xgb], axis = 0)
    result_xgb.to_csv(path + '/result_total_xgb_' + experiment_time + '.csv', index = False, sep = ',', encoding = 'UTF-8')
    
    ### 模型融合 采用普通加权
    result_ensemble = pd.DataFrame()
    
    result_ensemble['predicition'] = 0.5 * result_lgb['predicition'] + 0.5 * result_xgb['predicition']
    
    result_ensemble = pd.concat([result_lgb['id'], result_ensemble['predicition']], axis = 1)
    result_ensemble.to_csv(path + '/result_total_ensemble_' + experiment_time + '.csv', index = False, sep = ',', encoding = 'UTF-8')
    
    end_time = time.time()
    last = end_time - start_time
    print("total time used is %fmin and %fs" % (last // 60,  last % 60))