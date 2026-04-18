import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

import logging
from logging_code import setup_logging
logger=setup_logging('all_models')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.model_selection import GridSearchCV,cross_validate


def lr(x_train,x_test,y_train,y_test):
  global lr_reg
  global lr_pred
  lr_reg=LogisticRegression()
  lr_reg.fit(x_train,y_train)
  lr_pred=lr_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,lr_pred))
  logger.info(accuracy_score(y_test,lr_pred))
  logger.info(classification_report(y_test,lr_pred))
def dt(x_train,x_test,y_train,y_test):
  global dt_reg
  global dt_pred
  dt_reg=DecisionTreeClassifier(criterion='entropy')
  dt_reg.fit(x_train,y_train)
  dt_pred=dt_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,dt_pred))
  logger.info(accuracy_score(y_test,dt_pred))
  logger.info(classification_report(y_test,dt_pred))
def rf(x_train,x_test,y_train,y_test):
  global rf_reg
  global rf_pred
  rf_reg=RandomForestClassifier(criterion='entropy',n_estimators=5)
  rf_reg.fit(x_train,y_train)
  rf_pred=rf_reg.predict(x_test)
  logger.info(confusion_matrix(y_test,rf_pred))
  logger.info(accuracy_score(y_test,rf_pred))
  logger.info(classification_report(y_test,rf_pred))

def auc_roc_tech(x_train,y_train,x_test,y_test):
    lr_fpr, lr_tpr, lr_th = roc_curve(y_test, lr_pred)
    dt_fpr, dt_tpr, dt_th = roc_curve(y_test, dt_pred)
    rf_fpr, rf_tpr, rf_th = roc_curve(y_test, rf_pred)

    plt.figure(figsize=(5, 3))
    plt.plot([0, 1], [0, 1])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('all models auc curves')


    plt.plot(lr_fpr, lr_tpr, label='lr')
    plt.plot(dt_fpr, dt_tpr, label='dt')
    plt.plot(rf_fpr, rf_tpr, label='rf')

    plt.legend(loc=0)
    plt.show()

def hypertuning(x_train,y_train,x_test,y_test):
    try:
        parameters_list = [
            # L2
            {
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs'],
                'C': [0.1, 1, 10],
                'max_iter': [100, 200],
                'class_weight': [None, 'balanced']
            },

            #  L1
            {
                'penalty': ['l1'],
                'solver': ['liblinear'],
                'C': [0.1, 1, 10],
                'max_iter': [100, 200],
                'class_weight': [None, 'balanced']
            }
        ]
        grid_reg = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters_list, scoring='accuracy', cv=3)
        grid_result = grid_reg.fit(x_train, y_train)
        logger.info(f'The grid_result {grid_result}')
        logger.info(f'The grid best parameter are {grid_result.best_params_}')
        logger.info(f'The grid Accuracy Score {grid_result.best_score_}')

    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

def common(x_train,y_train,x_test,y_test):
    try:
        logger.info('-------lr----------')
        lr(x_train, x_test, y_train, y_test)
        logger.info('----dt-----')
        dt(x_train, x_test, y_train, y_test)
        logger.info('-----rf------')
        rf(x_train, x_test, y_train, y_test)
        logger.info(f'-----------auc_roc---------------')
        #auc_roc_tech(x_train,y_train,x_test,y_test)
        logger.info(f'*********HYPERPARAMETER--TUNING*******************************')
        hypertuning(x_train,y_train,x_test,y_test)
    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')
