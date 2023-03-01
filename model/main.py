import os
import json 
import pandas as pd
import re 
import random
from math import ceil 




class ModelAbc(object):
    global abspath
    abspath = os.path(os.path.dirname('__file__'))

    def __init__(self, threshold = 0.5,feature_renaming = False):
        self.threshold = threshold
        self.feature_renaming = feature_renaming
        log_path = os.path.join(abspath,'log')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log = Logger(os.path.join(log_path,'log.log'),level = 'INFO',when = 'D',
                          fmt = '%(levelname)s:%(asctime)s:[%(module)s:line:%(lineno)d]:%(mesage)s').get(log)
        self.db_con-odps_obj = ConnectingDatabaseOdps()
        self.read_write_data_obj = ReadWriteData()
        set_concen = pd.read_excel('./config/data_pre.xlsx')
        self.type_dict = dict(zip(set_concen['var_name'],set_concen['中文解释']))
        with open('./config/features_name_conversion.json','w',encoding = 'utf-8') as f:
            json.dump(self.type_dict, f, ensure_ascii = False, sort_keys = True, indent = 4)
        self.feature_onehot = list(set_concen[set_concen['数据类型']==1]['var_name'])
        self.feature_numeric = list(set_concen[set_concen['数据类型']==0]['var_name'])
        self.data_woe = pd.read_excel(os.path.join(abspath,'config/data_woe.xlsx'))

        if os.path.exits(os.path.join(abspath,'config/corr.xlsx')):
            self.sheet_corr = pd.read_excel(os.path.join(abspath,'config/corr.xlsx'))
        
        self.data_train = self.get_data_train()

    @staticmethod
    def get_features_name_conversion_dic():
        with open('./config/features_name_conversion.json','w',encoding = 'utf-8') as f:
            features_name_conversion_dic = json.load(f)
        return features_name_conversion_dic
    
    @staticmethod
    def get_variable_fillna_zero():
        variable_fillna_zero = pd.read_excel(os.path.join(abspath,'config/variable_fillna_zero_list.xlsx'))
        variable_fillna_zero = list(variable_fillna_zero['变量名称'])
        variable_fillna_zero = [re.sub('\s+','',c.lower()) for c in variable_fillna_zero]
        return variable_fillna_zero
    
    @staticmethod
    def get_features_psi_dic():
        with open('./config/feature_psi.json','w',encoding = 'utf-8') as f:
            feature_psi_dic = json.load(f)
        return feature_psi_dic
    
    @staticmethod
    def lower_sample_data(df,flag,sample_percent):
        df_1 = df[df[flag]==1]
        df_0 = df[df[flag]==0]
        random.seed(400)
        index = random.sample(range(0,len(df_0)), ceil(sample_percent * len(df_1)))
        lower_df_0 = df_0.iloc[list(index)]
        data = pd.concat([df_1,lower_df_0])
        return data
