import os
import sys
import warnings
from cmath import inf

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns

import logging
from logging_code import setup_logging
logger=setup_logging('main')

from Nulls import handling_nulls
from variable import vt
from categorical_to_num import c_t_n
from filter_methods import fm

from imblearn.over_sampling import SMOTE

from feature_scaling import fs

class TITANIC:
    def __init__(self,data):
        try:
            self.data=data
            self.df=pd.read_csv(self.data)
            self.df=self.df.drop(['Cabin','PassengerId','Ticket','Name'],axis=1)
            logger.info(f'shape {self.df.shape}')
            logger.info(f'sample data {self.df.sample(5)}')
            #logger.info(f'Data types {self.df.info()}')
            logger.info(f'missing values {self.df.isnull().sum()}')
            self.x=self.df.iloc[:,1:]
            self.y=self.df['Survived']
            #logger.info(self.x)
            #logger.info(self.y)
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=42)
            # print(f'pppp {self.y_train.dtype}')
            # print(f'pppp {self.y_test.dtype}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f"Error in line no : {error_line.tb_lineno} due to : {error_msg} type is {error_type}")

    def random_sample(self):
        try:
            logger.info(f'before train handling nulls {self.x_train.shape} \n {self.x_train.columns} \n {self.x_train.isnull().sum()}')
            logger.info(f'before test handling nulls {self.x_test.shape} \n {self.x_test.columns} \n {self.x_test.isnull().sum()}')
            self.x_train,self.x_test=handling_nulls(self.x_train,self.x_test)
            logger.info(f'before train handling nulls {self.x_train.shape} \n {self.x_train.columns} \n {self.x_train.isnull().sum()}')
            logger.info(f'before test handling nulls {self.x_test.shape} \n {self.x_test.columns} \n {self.x_test.isnull().sum()}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f"Error in line no : {error_line.tb_lineno} due to : {error_msg} type is {error_type}")

    def data_seperation(self):
        try:
            self.x_train_num_col=self.x_train.select_dtypes(exclude='object')
            self.x_test_num_col=self.x_test.select_dtypes(exclude='object')
            self.x_train_cat_col=self.x_train.select_dtypes(include='object')
            self.x_test_cat_col=self.x_test.select_dtypes(include='object')
            logger.info(f'train numeric columns {self.x_train_num_col.columns} \n {self.x_train_num_col.shape}')
            logger.info(f'test numeric columns {self.x_test_num_col.columns} \n {self.x_test_num_col.shape}')
            logger.info(f'train categorical columns {self.x_train_cat_col.columns} \n {self.x_train_cat_col.shape}')
            logger.info(f'test categorical columns {self.x_test_cat_col.columns} \n {self.x_test_cat_col.shape}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f"Error in line no : {error_line.tb_lineno} due to : {error_msg} type is {error_type}")

    def variable_transfor(self):
        try:
            logger.info(f'before variable transformation train columns {self.x_train_num_col.columns}')
            logger.info(f'before variable transformation test columns {self.x_test_num_col.columns}')
            self.x_train_num_col,self.x_test_num_col=vt(self.x_train_num_col,self.x_test_num_col)
            logger.info(f'after variable transformation train columns {self.x_train_num_col.columns}')
            logger.info(f'after variable transformation test columns {self.x_test_num_col.columns}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f"Error in line no : {error_line.tb_lineno} due to : {error_msg} type is {error_type}")

    def cat_to_num(self):
        try:
            self.x_train_cat_col,self.x_test_cat_col=c_t_n(self.x_train_cat_col,self.x_test_cat_col)
            # combine data
            self.x_train_num_col.reset_index(drop=True, inplace=True)
            self.x_train_cat_col.reset_index(drop=True, inplace=True)
            self.x_test_num_col.reset_index(drop=True, inplace=True)
            self.x_test_cat_col.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.x_train_num_col, self.x_train_cat_col], axis=1)
            self.testing_data = pd.concat([self.x_test_num_col, self.x_test_cat_col], axis=1)

            logger.info(f'========================================================================================')

            logger.info((f'final training data : {self.training_data.shape}'))
            logger.info((f'{self.training_data.columns}'))
            logger.info(f'training data null values : {self.training_data.isnull().sum()}')

            logger.info(f'final testing data : {self.testing_data.shape}')
            logger.info((f'{self.testing_data.columns}'))
            logger.info(f'testing data null values : {self.testing_data.isnull().sum()}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f"Error in line no : {error_line.tb_lineno} due to : {error_msg} type is {error_type}")

    def feature_selection(self):
        try:
            self.training_data,self.testing_data=f3m(self.training_data,self.testing_data,self.y_train,self.y_test)
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def data_balancing(self):
        try:
            logger.info(f'Number of Rows for GOOD customer {1} : {sum(self.y_train==1)}')
            logger.info(f'Number of Rows for BAD customer {0} : {sum(self.y_train==0)}')
            logger.info(f'Training data size : {self.training_data.shape}')

            sm = SMOTE(random_state=42)

            self.training_data_bal, self.y_train_bal = sm.fit_resample(self.training_data, self.y_train)

            logger.info(f'Number of Rows for GOOD customer {1} : {sum(self.y_train_bal == 1)}')
            logger.info(f'Number of Rows for BAD customer {0} : {sum(self.y_train_bal == 0)}')
            logger.info(f'Training data size : {self.training_data_bal.shape}')

            fs(self.training_data_bal, self.y_train_bal, self.testing_data, self.y_test)

        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')


if __name__=="__main__":
    try:
        obj=TITANIC('Titanic-Dataset.csv')
        obj.random_sample()
        obj.data_seperation()
        obj.variable_transfor()
        obj.cat_to_num()
        obj.feature_selection()
        obj.data_balancing()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f"Error in line no : {error_line.tb_lineno} due to : {error_msg} type is {error_type}")