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
logger=setup_logging('categorical_to_num')

from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

def c_t_n(x_train_cat,x_test_cat):
    try:
        logger.info(f'Before x_train_cat Columns {x_train_cat.shape} \n : {x_train_cat.columns}')
        logger.info(f'Befoer x_test_cat Columns {x_test_cat.shape} \n : {x_test_cat.columns}')
        oh = OneHotEncoder(drop='first')
        oh.fit(x_train_cat[['Sex', 'Embarked_randam']])
        value_train = oh.transform(x_train_cat[['Sex', 'Embarked_randam']]).toarray()
        value_test = oh.transform(x_test_cat[['Sex', 'Embarked_randam']]).toarray()

        t1 = pd.DataFrame(value_train)
        t2 = pd.DataFrame(value_test)
        t1.columns = oh.get_feature_names_out()
        t2.columns = oh.get_feature_names_out()
        x_train_cat.reset_index(drop=True, inplace=True)
        x_test_cat.reset_index(drop=True, inplace=True)
        t1.reset_index(drop=True, inplace=True)
        t2.reset_index(drop=True, inplace=True)
        x_train_cat = pd.concat([x_train_cat, t1], axis=1)
        x_test_cat = pd.concat([x_test_cat, t2], axis=1)
        x_train_cat = x_train_cat.drop(['Sex', 'Embarked_randam'], axis=1)
        x_test_cat = x_test_cat.drop(['Sex', 'Embarked_randam'], axis=1)
        logger.info(f'After NOMINAL x_train_cat Columns {x_train_cat.shape} \n : {x_train_cat.columns}')
        logger.info(f'After NOMINAL x_test_cat Columns {x_test_cat.shape} \n : {x_test_cat.columns}')

        return x_train_cat, x_test_cat

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f"Error in line no : {error_line.tb_lineno} due to : {error_msg} type is {error_type}")