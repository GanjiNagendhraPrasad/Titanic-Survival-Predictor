import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import warnings
warnings.filterwarnings("ignore")

import logging
from logging_code import  setup_logging
logger = setup_logging('Nulls')

def handling_nulls(x_train,x_test):
    try:
        logger.info(f'before train handling nulls {x_train.shape} \n {x_train.columns} \n {x_train.isnull().sum()}')
        logger.info(f'before test handling nulls {x_test.shape} \n {x_test.columns} \n {x_test.isnull().sum()}')
        for i in x_train.columns:
            if x_train[i].isnull().sum()>0:
                x_train[i+'_randam']=x_train[i].copy()
                x_test[i+'_randam']=x_test[i].copy()
                s=x_train[i].dropna().sample(x_train[i].isnull().sum(),random_state=42)
                s1=x_test[i].dropna().sample(x_test[i].isnull().sum(),random_state=42)
                s.index=x_train[x_train[i].isnull()].index
                s1.index=x_test[x_test[i].isnull()].index
                x_train.loc[x_train[i].isnull(), i+'_randam']=s
                x_test.loc[x_test[i].isnull(),i+'_randam']=s1
                x_train=x_train.drop([i],axis=1)
                x_test=x_test.drop([i],axis=1)
        logger.info(f'after train handling nulls {x_train.shape} \n {x_train.columns} \n {x_train.isnull().sum()}')
        logger.info(f'after test handling nulls {x_test.shape} \n {x_test.columns} \n {x_test.isnull().sum()}')
        return x_train,x_test

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f"Error in line no : {error_line.tb_lineno} due to : {error_msg} type is {error_type}")