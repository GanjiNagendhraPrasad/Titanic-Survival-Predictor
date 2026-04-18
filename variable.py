import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import warnings
warnings.filterwarnings("ignore")

import logging
from logging_code import  setup_logging
logger = setup_logging('variable')

import seaborn as sns
from scipy import stats
from scipy.stats import  boxcox
from scipy.stats import yeojohnson

def vt(x_train_num,x_test_num):
    try:
        logger.info(f'Before variable transformation train columns : {x_train_num.columns}')
        logger.info(f'Before variable transformation test columns : {x_test_num.columns}')
        # for i in x_train_num.columns:
        #     plt.figure(figsize=(8, 3))
        #     plt.subplot(1, 3, 1)
        #     plt.title('Outliers')
        #     sns.boxplot(x=x_train_num[i])
        #     plt.subplot(1, 3, 2)
        #     plt.title('Normal Distribution')
        #     x_train_num[i].plot(kind='kde')
        #     plt.subplot(1, 3, 3)
        #     plt.title('probplot')
        #     stats.probplot(x_train_num[i], dist='norm', plot=plt)
        #     plt.show()

        for i in x_train_num.columns:
            x_train_num[i+'_yeo'],lam_value=yeojohnson((x_train_num[i]))
            x_test_num[i+'_yeo'],lam_value=yeojohnson((x_test_num[i]))
            x_train_num=x_train_num.drop([i],axis=1)
            x_test_num=x_test_num.drop([i],axis=1)
            #TRIMING
            iqr = x_train_num[i + '_yeo'].quantile(0.75) - x_train_num[i + '_yeo'].quantile(0.25)
            upper_limit = x_train_num[i + '_yeo'].quantile(0.75) + (1.5 * iqr)
            lower_limit = x_train_num[i + '_yeo'].quantile(0.25) - (1.5 * iqr)

            x_train_num[i + '_trim'] = np.where(x_train_num[i + '_yeo'] > upper_limit, upper_limit,
                                                np.where(x_train_num[i + '_yeo'] < lower_limit, lower_limit,
                                                         x_train_num[i + '_yeo']))
            x_test_num[i + '_trim'] = np.where(x_test_num[i + '_yeo'] > upper_limit, upper_limit,
                                               np.where(x_test_num[i + '_yeo'] < lower_limit, lower_limit,
                                                        x_test_num[i + '_yeo']))
            x_train_num = x_train_num.drop([i + '_yeo'], axis=1)
            x_test_num = x_test_num.drop([i + '_yeo'], axis=1)

            # plt.figure(figsize=(5, 3))
            # sns.boxplot(x=x_train_num[i + '_trim'])
            # plt.show()

#caping with mean & std
            # lower_limit = x_train_num[i + '_yeo'].mean() - 3 * x_train_num[i + '_yeo'].std()
            # upper_limit = x_train_num[i + '_yeo'].mean() + 3 * x_train_num[i + '_yeo'].std()
            # x_train_num[i+'_cap']=np.where(x_train_num[i+'_yeo']>upper_limit,upper_limit,
            #                                 np.where(x_train_num[i+'_yeo']<lower_limit,lower_limit,x_train_num[i+'_yeo']))
            # x_test_num[i+'_cap']=np.where(x_test_num[i+'_yeo']>upper_limit,upper_limit,
            #                                np.where(x_test_num[i+'_yeo']<lower_limit,lower_limit,x_test_num[i+'_yeo']))
            # x_train_num = x_train_num.drop([i + '_yeo'], axis=1)
            # x_test_num = x_test_num.drop([i + '_yeo'], axis=1)
            #
            # plt.figure(figsize=(5, 3))
            # sns.boxplot(x=x_train_num[i + '_cap'])
            # plt.show()

#capping with 5th and 95th Quantile
            # lower_limit = x_train_num[i + '_yeo'].quantile(0.05)
            # upper_limit = x_train_num[i + '_yeo'].quantile(0.95)
            # x_train_num[i + '_cap'] = np.where(x_train_num[i + '_yeo'] > upper_limit, upper_limit,
            #                            np.where(x_train_num[i+'_yeo']<lower_limit,lower_limit,x_train_num[i+'_yeo']))
            # x_test_num[i+ '_cap']=np.where(x_test_num[i+'_yeo']>upper_limit,upper_limit,
            #                       np.where(x_test_num[i+'_yeo']<lower_limit,lower_limit,x_test_num[i+'_yeo']))
            #
            # x_train_num = x_train_num.drop([i + '_yeo'], axis=1)
            # x_test_num = x_test_num.drop([i + '_yeo'], axis=1)
            #
            # plt.figure(figsize=(5,3))
            # sns.boxplot(x=x_train_num[i+ '_cap'])
            # plt.show()

        logger.info(f'after variable transformation train columns {x_train_num.columns}')
        logger.info(f'after variable transformation test columns {x_test_num.columns}')

        return x_train_num, x_test_num

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f"Error in line no : {error_line.tb_lineno} due to : {error_msg} type is {error_type}")