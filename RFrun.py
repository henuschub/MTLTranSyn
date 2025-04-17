import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, cohen_kappa_score, mean_squared_error
import time
import random
from utils.tools import double_data, calculate

drugs = pd.read_csv('data/drug_features.csv')
print("drugs.shape:", drugs.shape)
cell_lines = pd.read_csv('data/cell_line_features.csv')
print("cell_lines.shape:", cell_lines.shape)
summary = pd.read_csv('data/oneil_summary_idx.csv')
print("summary.shape:", summary.shape)
FILE_URL = "result/RF_result.txt"


class DataLoader:
    def __init__(self, drugs, cell_lines, summary, test_fold, syn_threshold=30):
        self.drugs = drugs
        self.cell_lines = cell_lines
        self.summary = double_data(summary)
        self.syn_threshold = syn_threshold
        self.summary_test = self.summary.loc[self.summary['syn_fold'] == test_fold]
        self.summary_train = self.summary.loc[self.summary['syn_fold'] != test_fold]
        self.length_train = self.summary_train.shape[0]
        print("train:", self.length_train)
        self.length_test = self.summary_test.shape[0]
        print("test:", self.length_test)

    def syn_map(self, x):
        return 1 if x > self.syn_threshold else 0

    def get_samples(self, flag, method):
        if flag == 0:  # train data
            summary = self.summary_train
        else:  # test data
            summary = self.summary_test
        d1_idx = summary.iloc[:, 0]
        d2_idx = summary.iloc[:, 1]
        c_idx = summary.iloc[:, 2]
        d1 = np.array(self.drugs.iloc[d1_idx])
        d2 = np.array(self.drugs.iloc[d2_idx])
        c_exp = np.array(self.cell_lines.iloc[c_idx])
        X = np.concatenate((d1, d2, c_exp), axis=1)
        if method == 0:  # regression
            y = np.array(summary.iloc[:, 5])
        else:  # classification
            y = np.array(summary.iloc[:, 5].apply(lambda s: self.syn_map(s)))
        return X, y


Fold = 5
print("----------- Classification ----------")
with open(FILE_URL, 'a') as file:
    file.write("---------------------- Classification ---------------------\n")
result_c = []
for fold_test in range(0, Fold):
    print("---------- Test Fold " + str(fold_test) + " ----------")
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    random.seed(1)
    np.random.seed(1)
    with open(FILE_URL, 'a') as file:
        file.write("---------- Test Fold " + str(fold_test) + " ----------\n")
        file.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + "\n")
    sampelData = DataLoader(drugs, cell_lines, summary, test_fold=fold_test)
    x_train, y_train = sampelData.get_samples(0, 1)
    x_test, syn_true_label = sampelData.get_samples(1, 1)
    hyper_params = {'n_estimators': [128, 512, 1024], 'max_features': ['sqrt', 256, 512]}
    gbc = RandomForestClassifier(max_depth=20, min_samples_split=100, min_samples_leaf=20, random_state=1,
                                 class_weight={0: 1, 1: 5})
    grid_cv = GridSearchCV(gbc, param_grid=hyper_params, scoring='roc_auc', verbose=10, cv=4)

    grid_cv.fit(x_train, y_train)
    syn_pred_label = grid_cv.predict(x_test)
    syn_pred_prob = grid_cv.predict_proba(x_test)
    n = sampelData.length_test // 2
    syn_true_label = syn_true_label[0:n]
    syn_pred_label = syn_pred_label[0:n]
    syn_pred_prob = (syn_pred_prob[0:n, ] + syn_pred_prob[n:, ]) / 2
    syn_pred_prob = syn_pred_prob[:, 1]
    syn_prec, syn_recall, syn_threshold = precision_recall_curve(syn_true_label, syn_pred_prob)
    syn_TP = np.sum(np.logical_and(syn_pred_label, syn_true_label))
    syn_FP = np.sum(np.logical_and(syn_pred_label, np.logical_not(syn_true_label)))
    syn_TN = np.sum(np.logical_and(np.logical_not(syn_pred_label), np.logical_not(syn_true_label)))
    syn_FN = np.sum(np.logical_and(np.logical_not(syn_pred_label), syn_true_label))
    syn_metrics = {}
    syn_metrics["ROC AUC"] = roc_auc_score(syn_true_label, syn_pred_prob)
    syn_metrics["PR AUC"] = auc(syn_recall, syn_prec)
    syn_metrics["ACC"] = (syn_TP + syn_TN) / (syn_TP + syn_FP + syn_TN + syn_FN)
    syn_metrics["TPR"] = syn_TP / (syn_TP + syn_FN)
    syn_metrics["TNR"] = syn_TN / (syn_TN + syn_FP)
    syn_metrics["BACC"] = (syn_metrics["TPR"] + syn_metrics["TNR"]) / 2
    syn_metrics["PREC"] = syn_TP / (syn_TP + syn_FP)
    syn_metrics["Kappa"] = cohen_kappa_score(syn_true_label, syn_pred_label)
    result_c.append(syn_metrics)
    print("syn_metrics:", syn_metrics)
    with open(FILE_URL, 'a') as file:
        file.write("syn_metrics:" + str(syn_metrics) + "\n")
calculate(np.array(result_c), "classification", Fold, FILE_URL)

print("----------- Regression ----------")
with open(FILE_URL, 'a') as file:
    file.write("---------------------- Regression ---------------------\n")
result_r = []
for fold_test in range(0, Fold):
    print("---------- Test Fold " + str(fold_test) + " ----------")
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    random.seed(1)
    np.random.seed(1)
    with open(FILE_URL, 'a') as file:
        file.write("---------- Test Fold " + str(fold_test) + " ----------\n")
        file.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + "\n")
    sampelData = DataLoader(drugs, cell_lines, summary, test_fold=fold_test)
    x_train, y_train = sampelData.get_samples(0, 0)
    x_test, syn_true_value = sampelData.get_samples(1, 0)
    hyper_params = {'n_estimators': [128, 512, 1024], 'max_features': ['sqrt', 256, 512]}
    gbr = RandomForestRegressor(max_depth=20, min_samples_split=100, min_samples_leaf=20, random_state=1)
    grid_cv = GridSearchCV(gbr, param_grid=hyper_params, scoring='neg_mean_squared_error', verbose=10, cv=4)

    grid_cv.fit(x_train, y_train)
    syn_pred_value = grid_cv.predict(x_test)
    n = sampelData.length_test // 2
    syn_true_value = syn_true_value[0:n]
    syn_pred_value = (syn_pred_value[0:n] + syn_pred_value[n:]) / 2
    syn_metrics = {}
    syn_metrics['MSE'] = mean_squared_error(syn_true_value, syn_pred_value)
    syn_metrics['RMSE'] = np.sqrt(syn_metrics['MSE'])
    syn_metrics["Pearsonr"] = pearsonr(syn_true_value, syn_pred_value)[0]
    result_r.append(syn_metrics)
    print("syn_metrics:", syn_metrics)
    with open(FILE_URL, 'a') as file:
        file.write("syn_metrics:" + str(syn_metrics) + "\n")
calculate(np.array(result_r), "regression", Fold, FILE_URL)
