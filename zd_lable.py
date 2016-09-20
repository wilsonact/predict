#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import (print_function, unicode_literals)

from sklearn.linear_model import (LinearRegression, Ridge, 
                  Lasso, RandomizedLasso, LogisticRegression)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from minepy import MINE
import MySQLdb
import numpy as np
import pandas as pd
import pylab
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets, metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import BernoulliRBM
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from itertools import cycle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
# from sklearn.ensemble import RandomForestRegressor

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

__author__ = 'Wilson'

'Data procedure'
def read_table(sql_col_list, l1_num, l2_num):
    print('Loading data...')
    conn = MySQLdb.connect(
        host='172.30.115.62', port=3306, user='pmd',
        passwd='Iccc2016datapmd', db='pmd')
    # sql1 = 'SELECT amount_usd_this3mon,amount_rmb_this3mon,ss_pt_amount_all_thismon,ss_pt_amount_all_pre1mon,ss_pt_amount_all_pre4mon,ss_pt_amount_all_pre2mon,na_cust_nbr,bs_crlimit,ss_pt_amount_all_pre3mon,bs_int_bnp,age,epp_limit_status,love_yyh,huge_pay,bs_avl_bal,epp_rush_amounts,home_care_times,bs_cash_bal_bnp,ss_pt_amount_wholesale_thismon,ss_pt_amount_digital_pre5mon,deferd_day,love_car,ss_pt_amount_wholesale_pre2mon,ss_pt_amount_wholesale_pre1mon,hb_fh_percent_14entire,ss_pt_amount_all_pre5mon,bs_mp_bal_bnp,no_anypay_status,ss_pt_amount_life_pre1mon,ss_pt_amount_life_pre3mon,ss_pt_amount_life_thismon,ss_pt_amount_wholesale_pre3mon,ss_pt_amount_life_pre2mon,ss_pt_amount_wholesale_pre4mon,ss_pt_amount_digital_thismon,ss_pt_amount_life_pre4mon,ss_pt_amount_cinema_pre5mon,ss_pt_amount_digital_pre1mon,ss_pt_amount_digital_pre2mon,cash_amounts,ss_pt_amount_digital_pre3mon,ss_pt_amount_dailyuse_pre1mon,ss_pt_amount_digital_pre4mon,ss_pt_amount_fin_pre5mon,ss_pt_amount_dailyuse_thismon,ss_pt_amount_dailyuse_pre2mon,bs_retail_bal_bnp,ss_pt_amount_dailyuse_pre4mon,ss_pt_amount_dailyuse_pre3mon,his_sus_prom_times,ss_pt_amount_food_pre5mon,ss_pt_amount_dailyuse_pre5mon,ss_pt_amount_hotel_pre5mon,ss_pt_amount_food_pre1mon,tv_amounts,line_incr,ss_pt_amount_food_thismon,ss_pt_amount_food_pre2mon,ss_pt_amount_mall_thismon,other_province,ss_pt_amount_food_pre4mon,ss_pt_amount_mall_pre1mon,ss_pt_amount_business_pre1mon,ss_pt_amount_business_thismon,net_pos_times FROM tm_train where label = 0 limit 150000 ;'

    sql1 = 'SELECT ' +  sql_col_list + ' FROM ZD_07_train where amount_rmb_this3mon>0 and label = 0 limit ' + str(l1_num) +';'
    sql2 = 'SELECT ' +  sql_col_list + ' FROM ZD_07_train where amount_rmb_this3mon>0 and label = 1 limit ' + str(l2_num) +';'

    df1 = pd.read_sql_query(sql1, conn)
    df1 = df1.replace('NULL', 0)
    df1 = df1.replace(' ', 0)
    df1 = df1.fillna(0)

    df2 = pd.read_sql_query(sql2, conn)
    df2 = df2.replace('NULL', 0)
    df2 = df2.replace(' ', 0)
    df2 = df2.fillna(0)

    df = pd.concat([df1, df2])
    conn.close()

    return df
def select_test(sql_col_list, n):
    print('Loading test data...')
    conn = MySQLdb.connect(
        host='172.30.115.62', port=3306, user='pmd',
        passwd='Iccc2016datapmd', db='pmd')

    # sql3 = 'SELECT amount_usd_this3mon,amount_rmb_this3mon,ss_pt_amount_all_thismon,ss_pt_amount_all_pre1mon,ss_pt_amount_all_pre4mon,ss_pt_amount_all_pre2mon,na_cust_nbr,bs_crlimit,ss_pt_amount_all_pre3mon,bs_int_bnp,age,epp_limit_status,love_yyh,huge_pay,bs_avl_bal,epp_rush_amounts,home_care_times,bs_cash_bal_bnp,ss_pt_amount_wholesale_thismon,ss_pt_amount_digital_pre5mon,deferd_day,love_car,ss_pt_amount_wholesale_pre2mon,ss_pt_amount_wholesale_pre1mon,hb_fh_percent_14entire,ss_pt_amount_all_pre5mon,bs_mp_bal_bnp,no_anypay_status,ss_pt_amount_life_pre1mon,ss_pt_amount_life_pre3mon,ss_pt_amount_life_thismon,ss_pt_amount_wholesale_pre3mon,ss_pt_amount_life_pre2mon,ss_pt_amount_wholesale_pre4mon,ss_pt_amount_digital_thismon,ss_pt_amount_life_pre4mon,ss_pt_amount_cinema_pre5mon,ss_pt_amount_digital_pre1mon,ss_pt_amount_digital_pre2mon,cash_amounts,ss_pt_amount_digital_pre3mon,ss_pt_amount_dailyuse_pre1mon,ss_pt_amount_digital_pre4mon,ss_pt_amount_fin_pre5mon,ss_pt_amount_dailyuse_thismon,ss_pt_amount_dailyuse_pre2mon,bs_retail_bal_bnp,ss_pt_amount_dailyuse_pre4mon,ss_pt_amount_dailyuse_pre3mon,his_sus_prom_times,ss_pt_amount_food_pre5mon,ss_pt_amount_dailyuse_pre5mon,ss_pt_amount_hotel_pre5mon,ss_pt_amount_food_pre1mon,tv_amounts,line_incr,ss_pt_amount_food_thismon,ss_pt_amount_food_pre2mon,ss_pt_amount_mall_thismon,other_province,ss_pt_amount_food_pre4mon,ss_pt_amount_mall_pre1mon,ss_pt_amount_business_pre1mon,ss_pt_amount_business_thismon,net_pos_times FROM tm_test  limit ' + str(n) + ',50000;'
    sql3 = 'SELECT ' +sql_col_list + ' FROM ZD_07_test where amount_rmb_this3mon>0 limit ' + \
        str(n) + ',150000;'

    df_all = pd.read_sql_query(sql3, conn)
    df_all = df_all.replace('NULL', 0)
    df_all = df_all.replace(' ', 0)
    df_all = df_all.fillna(0)

    conn.close()
    print(df_all.shape, 'data has been loaded.')
    return df_all

'Feature'
def select_features(df, threshold=0.0):
    # sf = VarianceThreshold(threshold=threshold)
    # X = sf.fit_transform(X_all)
    X = df.iloc[:, 0:36]
    label = df.columns[-1]
    # print(X.columns)
    Y = df[label]

    return X, Y

'Train'
def random_forest_train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=None, test_size=None, train_size=0.9)
    # clf = RandomForestRegressor()
    clf = RandomForestClassifier(
        n_estimators=200, criterion='gini', max_features='auto',
        max_depth=None, bootstrap='True', n_jobs=4)
    clf.fit(X_train, y_train)   
    return clf, X_test, y_test
def gbdt_train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=None, test_size=None, train_size=0.9)
    # clf = RandomForestRegressor()
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01,
                                     max_depth=6, random_state=0)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test
def decision_tree_train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=None, test_size=None, train_size=0.9)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf, X_test, y_test
def log_train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=None, test_size=None, train_size=0.9)
    clf = LogisticRegression(C=100, penalty='l1', tol=0.01)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test
def nb_train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=None, test_size=None, train_size=0.9)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf, X_test, y_test

def nnw_train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=None, test_size=None, train_size=0.9)

    logistic = linear_model.LogisticRegression(C=100, penalty='l1', tol=0.01)
    rbm = BernoulliRBM(random_state=0, verbose=True)

    clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    clf.fit(X_train, y_train)
    return clf, X_test, y_test

'Test'
def predict_test(clf, X_test, y_test):
    p1 = clf.predict_proba(X_test)
    p1 =  p1[:, 1]
    preds = clf.predict(X_test)
    # print (clf.get_params())
    tab = pd.crosstab(
        y_test, preds, rownames=['actual'], colnames=['Validation'])

    report = classification_report(y_test, preds)
    print(tab)
    # print('tab3')
    # print(tab3)
    print(report)
    # csv
    # y_test.to_csv('x.csv')
    # np.savetxt("p1.csv", p1[:, 1], delimiter=",")
    # Plot ROC
    anypay_score = X_test.iloc[:, 35:36].astype(float)
    # print (X_test.shape)

    # y_score = clf.score(X_test, y_test)
    # plot_roc(y_test, p1, anypay_score)
    # plot_roc(y_test, p1[:, 1])
  
    return tab, y_test, p1, anypay_score

def plot_roc(y_true, y_pred, y_score):
    ax = pylab.subplot(1, 1, 1)

    fpr, tpr, th = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    fpr1, tpr1, th1 = roc_curve(y_true, y_pred)
    print (metrics.auc(fpr1, tpr1))
    ax.plot(fpr1, tpr1)
    ax.set_title('ROC Curve')
    pylab.grid(True)

    pylab.show()

'Feature_score'
def rank_to_dict(ranks, names, order=1):
  minmax = MinMaxScaler()
  ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
  ranks = map(lambda x: round(x, 2), ranks)
  return dict(zip(names, ranks ))
def feature_scoring(X, Y):
    names = ["x%s" % i for i in range(1,37)]
    ranks = {}
    
    X = X.values[:, :]
    lr = LinearRegression(normalize=True)
    lr.fit(X, Y)
    ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)
    
    ridge = Ridge(alpha=7)
    ridge.fit(X, Y)
    ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)
    
    lasso = Lasso(alpha=.05)
    lasso.fit(X, Y)
    ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)
    
    rlasso = RandomizedLasso(alpha=0.04)
    rlasso.fit(X, Y)
    ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)
    
    #stop the search when 5 features are left (they will get equal scores)
    rfe = RFE(lr, n_features_to_select=5)
    rfe.fit(X,Y)
    ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)
    
    rf = RandomForestRegressor()
    rf.fit(X,Y)
    ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
    
    f, pval  = f_regression(X, Y, center=True)
    ranks["Corr."] = rank_to_dict(f, names)
    
    print('startMIC')
    mine = MINE()
    mic_scores = []
    
    for i in range(X.shape[1]):
      mine.compute_score(X[:,i], Y)
      m = mine.mic()
      mic_scores.append(m)
      print(i)
    ranks["MIC"] = rank_to_dict(mic_scores, names)
    
    print('finish MIc')
      
    r = {}
    for name in names:
      r[name] = round(np.mean([ranks[method][name] 
                   for method in ranks.keys()]), 2)
    methods = sorted(ranks.keys())
    ranks["Mean"] = r
    methods.append("Mean")
    
    print ("\t%s" % "\t".join(methods))
    for name in names:
      print ("%s\t%s" % (name, "\t".join(map(str, 
                 [ranks[method][name] for method in methods]))))


'''Run'''
#Data
sql_col_list = 'ss_pt_amount_all_pre2mon,'\
               'ss_pt_amount_all_thismon/bs_crlimit,'\
               'amount_rmb_this3mon,'\
               'ss_pt_amount_all_thismon/bs_crlimit,'\
               'bs_avl_bal,'\
               'amount_rmb_this3mon/bs_crlimit,'\
               'ss_pt_amount_all_pre1mon/bs_crlimit,'\
               'bs_retail_bal_bnp,'\
               'bs_crlimit,'\
               'ss_pt_amount_all_pre2mon/bs_crlimit,'\
               'ss_pt_amount_all_pre4mon/bs_crlimit,'\
               'ss_pt_amount_all_pre3mon/bs_crlimit,'\
               'ss_pt_amount_wholesale_thismon,'\
               'ss_pt_amount_wholesale_pre1mon/bs_crlimit,'\
               'ss_pt_amount_wholesale_pre2mon/bs_crlimit,'\
               'ss_pt_amount_life_pre1mon/bs_crlimit,'\
               'ss_pt_amount_life_pre2mon/bs_crlimit,'\
               'ss_pt_amount_life_pre3mon,'\
               'ss_pt_amount_digital_thismon,'\
               'ss_pt_amount_life_pre4mon,'\
               'ss_pt_amount_digital_pre2mon,'\
               'ss_pt_amount_digital_pre3mon,'\
               'ss_pt_amount_digital_pre4mon,'\
               'ss_pt_amount_digital_pre1mon,'\
               'ss_pt_amount_food_thismon,'\
               'ss_pt_amount_mall_thismon,'\
               'ss_pt_amount_food_pre1mon,'\
               'ss_pt_amount_food_pre2mon,'\
               'ss_pt_amount_business_thismon,'\
               'ss_pt_amount_food_pre3mon,'\
               'ss_pt_amount_hotel_thismon,'\
               'ss_pt_amount_mall_pre2mon,'\
               'ss_pt_amount_mall_pre1mon,'\
               'ss_pt_amount_mall_pre3mon,'\
               'ss_pt_amount_hotel_pre2mon,'\
               'score_anypay,'\
               'label'

df = read_table(sql_col_list, 16000, 16000)
X_tr, Y_tr = select_features(df)
del df

#Feature
# feature_scoring(X_tr, Y_tr)    

#Training Classifiers

clf_rf,   X_v_rf,   Y_v_rf  = random_forest_train(X_tr, Y_tr)
# joblib.dump(clf_rf,'./rf_m03_1600.m', compress = 3)
# clf_rf = joblib.load('./rf_m04.m')
clf_gbdt, X_v_gbdt, Y_v_gbdt = gbdt_train(X_tr, Y_tr)
clf_dt,   X_v_dt,   Y_v_dt   = decision_tree_train(X_tr, Y_tr)
clf_log,  X_v_log,  Y_v_log  = log_train(X_tr, Y_tr)
clf_nb,   X_v_nb,   Y_v_nb   = nb_train(X_tr, Y_tr)
clf_nnw,  X_v_nnw,  Y_v_nnw  = nnw_train(X_tr, Y_tr)

ax = pylab.subplot(1, 1, 1)
ax.set_title('ROC Curve')


# auc area
# print (metrics.auc(fpr1, tpr1))



# Validate the rest of Sample Data

# tab, y_test, p1, anypay_score = predict_test(clf_rf,  X_v_rf,   Y_v_rf )
# print(tab)
# fpr, tpr, th = roc_curve(y_test, p1)
# ax.plot(fpr, tpr, label = 'rf')
# fpr, tpr, th = roc_curve(y_test, anypay_score)
# ax.plot(fpr, tpr, label = 'anypay_score')

# tab, y_test, p1, anypay_score = predict_test(clf_gbdt,X_v_gbdt, Y_v_gbdt)
# print (tab)
# fpr, tpr, th = roc_curve(y_test, p1)
# ax.plot(fpr, tpr, label = 'gbdt')

# tab, y_test, p1, anypay_score = predict_test(clf_dt,  X_v_dt,   Y_v_dt)
# print (tab)
# fpr, tpr, th = roc_curve(y_test, p1)
# ax.plot(fpr, tpr, label = 'dt')

# tab, y_test, p1, anypay_score = predict_test(clf_svm, X_v_svm,  Y_v_svm)
# print (tab)
# fpr, tpr, th = roc_curve(y_test, p1)
# ax.plot(fpr, tpr,label = 'svm')
# ax.legend(loc='best')
# print('test finished')
# pylab.grid(True)
# pylab.show()


# Predict and Fit
# clear ram
# del X_tr, Y_tr, df_all
n = 15

# test data
df_all = select_test(sql_col_list, 3000000)
X_test, Y_test = select_features(df_all)

#Plot ROC

tab, y_test, p1, anypay_score = predict_test(clf_rf, X_test, Y_test )
fpr, tpr, th = roc_curve(y_test, p1)
ax.plot(fpr, tpr, label = 'RF')
print('RandomF \nAUC:')
print (metrics.auc(fpr, tpr))
fpr, tpr, th = roc_curve(y_test, anypay_score)
ax.plot(fpr, tpr, label = 'anypay_score')
print (metrics.auc(fpr, tpr))
# tab, y_test, p1, anypay_score = predict_test(clf_gbdt, X_test, Y_test)
# fpr, tpr, th = roc_curve(y_test, p1)
# print ('GBDT\nAUC:')
# print (metrics.auc(fpr, tpr))
# ax.plot(fpr, tpr, label = 'GBDT')

tab, y_test, p1, anypay_score = predict_test(clf_dt, X_test, Y_test)
fpr, tpr, th = roc_curve(y_test, p1)
print ('decison tree \nAUC:')
print (metrics.auc(fpr, tpr))
ax.plot(fpr, tpr, label = 'DT')

tab, y_test, p1, anypay_score = predict_test(clf_log, X_test, Y_test)
fpr, tpr, th = roc_curve(y_test, p1)
print ('logistic \nAUC:')
print (metrics.auc(fpr, tpr))
ax.plot(fpr, tpr,label = 'Logistic')
ax.legend(loc='best')

# tab, y_test, p1, anypay_score = predict_test(clf_nb, X_test, Y_test)
# print ('NB')
# fpr, tpr, th = roc_curve(y_test, p1)
# ax.plot(fpr, tpr,label = 'NB')
# ax.legend(loc='best')
# print('test finished')

'''
tab, y_test, p1, anypay_score = predict_test(clf_nnw, X_test, Y_test)
print ('NNw')
fpr, tpr, th = roc_curve(y_test, p1)
ax.plot(fpr, tpr,label = 'NNw')
ax.legend(loc='best')
print('test finished')
'''
pylab.grid(True)
pylab.show()

'''
# while (n * 20000 < 11946144):
while (n * 150000 < 12000000):
    df_all = select_test(sql_col_list, n * 150000)
    X_test, Y_test = select_features(df_all)
    tab = decision_tree_test(clf, X_test, Y_test)
    tab_all = tab_all + tab
    n = n + 1
    del df_all, X_test, Y_test, tab
    print(n)
    print(tab_all)
'''
