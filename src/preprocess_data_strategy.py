# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 07:50:31 2020

@author: Victolo Porsche
"""

from preprocess_data import PreprocessData

class PreprocessDataStrategy():
    '''
    This completes the preprocessing of the input data
    '''
    
    def __init__(self):
        
        self.data = None
        self._preprocessor = PreprocessData()
        
        
    def strategy(self, data, strategy_type="strategy1"):
        self.data=data
        if strategy_type=='strategy1':
            self._strategy1()
        elif strategy_type=='strategy2':
            self._strategy2()

        return self.data
    
    def _base_strategy(self):
        #drop_strategy = {'': 1,  # 1 indicate axis 1(column)
                         #'': 1,
                         #'': 1}
        #self.data = self._preprocessor.drop(self.data, drop_strategy)

        fill_strategy = {'oldbalanceOrg': 'Median',
                         'newbalanceOrig': 'Median',
                         'oldbalanceDest': 'Mode',
                         'newbalanceDest': 'Mode',
                         'amount': 'Mode'}
        self.data = self._preprocessor.fillna(self.data, fill_strategy)
        self.data = self._preprocessor.feature_engineering(self.data, 1)
        self.data = self._preprocessor._label_encoder(self.data)
        self.data = self._preprocessor._min_max_scaler(self.data)

        

        
    def _strategy1(self):
        self._base_strategy()

        self.data=self._preprocessor._get_dummies(self.data,
                                        prefered_columns=['type'])
        
    def _strategy2(self):
        self._base_strategy()

        self.data=self._preprocessor._get_dummies(self.data,
                                        prefered_columns=None)

        
        