# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:40:29 2020

@author: Victolo Porsche
"""
from data_information import DataInformation
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import csv

class PreprocessData():
    '''
    This is a class which preprocess the raw data 
    '''
    
    def __init__(self):
        print('Preprocess Object Created')
        
        self._info=DataInformation()
        
    def fillna(self, data, fill_strategies):
        for column, strategy in fill_strategies.items():
            if strategy == 'None':
                data[column] = data[column].fillna('None')
            elif strategy == 'Zero':
                data[column] = data[column].fillna(0)
            elif strategy == 'Mode':
                data[column] = data[column].fillna(data[column].mode()[0])
            elif strategy == 'Mean':
                data[column] = data[column].fillna(data[column].mean())
            elif strategy == 'Median':
                data[column] = data[column].fillna(data[column].median())
            else:
                print('{}: There is no such preprocessing strategy available'.format(strategy))
                
        return data
    
    def drop(self, data, drop_strategies):
        for column, strategy in drop_strategies.items():
            data = data.drop(labels=[column], axis=strategy)
            
        return data
    
    def feature_engineering(self, data, engineering_strategies=1):
        if engineering_strategies==1:
            return self._feature_engineering1(data)
        return data
    
    def _feature_engineering1(self,data):
        data = self._base_feature_engineering(data)
        
        data['AmountBin'] = pd.qcut(data['amount'], 4)
        data['OldbalanceOrgBin'] = pd.qcut(data['oldbalanceOrg'], 4, duplicates='drop')
        data['NewbalanceOrigBin'] = pd.qcut(data['newbalanceOrig'], 4, duplicates='drop')
        data['OldbalanceDestBin'] = pd.qcut(data['oldbalanceDest'], 4, duplicates='drop')
        data['NewbalanceDestBin'] = pd.qcut(data['newbalanceDest'], 4, duplicates='drop')
        
        drop_strategy = {'amount': 1,
                         'nameOrig': 1,
                         'oldbalanceOrg': 1,
                         'newbalanceOrig': 1,
                         'oldbalanceDest': 1,
                         'newbalanceDest': 1}
                
        data = self.drop(data, drop_strategy)
        
        return data
    
    def _base_feature_engineering(self,data):
        data['DifferenceBalanceOrg'] = data['newbalanceOrig'] - data['oldbalanceOrg']
        data['DifferenceBalanceDesc'] = data['newbalanceDest'] - data['oldbalanceDest']
        
        return data
    
    def _label_encoder(self,data):
        labelEncoder=LabelEncoder()
        for column in data.columns.values:
            if 'int64'==data[column].dtype or 'float64'==data[column].dtype or 'int64'==data[column].dtype:
                continue
            labelEncoder.fit(data[column])
            data[column]=labelEncoder.transform(data[column])
        return data

    def _min_max_scaler(self, data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit(data)
        scaled_data = scaler.transform(data)
        return data

    
    def mapping_dict(self, data):
        labelEncoder=LabelEncoder()
        encoded_dict = {}
        categorical_data = self._info.categorical_features(data)
        for feature in categorical_data:
            data[feature] = labelEncoder.fit_transform(data[feature])
            categorical_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
            encoded_dict[feature]=categorical_mapping
            
        return encoded_dict
    
    def save_mappings(self, data):
        encoded_dict = self.mapping_dict(data)
        
        with open('../output/dict_titanic.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in encoded_dict.items():
                writer.writerow([key, value])
                
    def _get_dummies(self, data, prefered_columns=None):

        if prefered_columns is None:
            columns=data.columns.values
            non_dummies=None
        else:
            non_dummies=[col for col in data.columns.values if col not in prefered_columns ]

            columns=prefered_columns


        dummies_data=[pd.get_dummies(data[col],prefix=col) for col in columns]

        if non_dummies is not None:
            for non_dummy in non_dummies:
                dummies_data.append(data[non_dummy])

        return pd.concat(dummies_data, axis=1)
        
            

        
