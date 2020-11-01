# -*- coding: utf-8 -*-

"""
Created on Sat Oct 17 14:03:16 2020

@author: Victolo Porsche
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DataInformation():
    '''
    This class prints the summary information
    about the classification data on the screen
    '''
    
    def __init__(self):
        
        '''
        Prints a brief information about the dataset
        '''
        print('Information object created')
        
    def _get_missing_values(self, data):
        '''
        find any missing values in the dataset
        '''
        missing_values = data.isnull().sum()
        missing_values.sort_values(ascending=False, inplace=True)
        
        return missing_values
    
    def _data_description(self, data):
        
        data_description = data.describe()
        
        return data_description
    
    def info(self, data):
        '''
        prints feature name, datatype, number of missing values and all samples of each feature
        : param data: dataset information will be gathered from 
        :return: no return value
        '''
        feature_dtypes= data.dtypes
        self.missing_values = self._get_missing_values(data)
        
        print('=' * 50)
        
        print('{:16} {:16} {:25} {:16}'.format('Feature Name'.upper(), 'Data Format'.upper(),
              'Number of Missing Values'.upper(), 'Samples'.upper()))
        
        for feature_name, dtype, missing_value in zip(self.missing_values.index.values,
                                                      feature_dtypes[self.missing_values.index.values],
                                                      self.missing_values.values):
            print('{:18} {:19} {:19}'.format(feature_name, str(dtype), str(missing_value)), end='')
            for v in data[feature_name].values[:10]:
                print(v, end=',')
                print()
                print('='*50)
                
    def number_missing(self, data):
        
        features_with_na = [features for features in data.columns if data[features].isnull().sum()>0]
        for feature in features_with_na:
            print(feature, np.round(data[feature].isnull().sum(), 4), 'missing values')
            
            return features_with_na
            
        else:
            print("There are no missing values in this dataset")
            
    def numerical_features(self, data):
        numerical_features = [feature for feature in data.columns if data[feature].dtype != 'O']
        print('Number of numerical features:', len(numerical_features))
        numerical_features_data = data[numerical_features]
        
        return numerical_features_data
    
    def categorical_features(self, data):
        categorical_features = [feature for feature in data.columns if data[feature].dtypes=='O']
        print('Number of categorical features:', len(categorical_features))
        categorical_features_data = data[categorical_features]
        
        return categorical_features_data
    
    def discrete_features(self, data):
        numerical_features = self.numerical_features(data)
        discrete_features = [feature for feature in numerical_features if len(data[feature].unique())<25 and 
                     feature not in ['id'] + ['isFraud']]
        print('Discrete features count: {}'.format(len(discrete_features)))
        discrete_features_data = data[discrete_features]
        
        return discrete_features_data
    
    def continous_features(self, data):
        numerical_features = self.numerical_features(data)
        continous_feature= [feature for feature in numerical_features if len(data[feature].unique())>25 and
                            feature not in ['id'] + ['isFraud']]
        print("continous feature count {}". format(len(continous_feature)))
        continous_features_data = data[continous_feature]
        
        return continous_features_data
    
    def discrete_plot(self, data):
        discrete_data = self.discrete_features(data)
        
        for feature in discrete_data:
            
            data.groupby(feature)['isFraud'].count().plot.bar()
            plt.xlabel(feature)
            plt.ylabel('isFraud')
            plt.title(feature)
            plt.show()
        else:
            print('No discrete featrues, hence no plot available')
            
    
    def continous_plot(self, data):
        continous_data = self.continous_features(data)
        
        for feature in continous_data:
            
            data.groupby(feature)['isFraud'].count().plot.bar()
            data[feature].hist(bins=25)
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.title(feature)
            plt.show()
            
            pass 
        else:
            print('No continous variable, hence no plot available')
            
    def discrete_detect_outliers(self, data):
        discrete_data = self.discrete_features(data)
        
        for feature in discrete_data:
            
            if 0 in data[feature].unique():
                pass
            else:
                data[feature]=np.log(data[feature])
                data.boxplot(column=feature)
                plt.ylabel(feature)
                plt.title(feature)
                plt.show()
                
    def continous_detect_outliers(self, data):
        continous_data = self.continous_features(data)
        
        for feature in continous_data:
            
            
            data[feature]=np.log(data[feature])
            data.boxplot(column=feature)
            plt.ylabel(feature)
            plt.title(feature)
            plt.show()
       
    def unique_categorical(self, data):
        categorical_features = self.categorical_features(data)
        for feature in categorical_features:
            print('The feature {} and the number of unique categories are {}'
                  .format(feature,len(data[feature].unique())))
         
    def plot_categorical_dependent(self, data):
        categorical_features = self.categorical_features(data)
        
        for feature in categorical_features:
            
            data.groupby(feature)['isFraud'].count().plot.bar()
            plt.xlabel(feature)
            plt.ylabel('isFraud')
            plt.title(feature)
            plt.show()
        
      
