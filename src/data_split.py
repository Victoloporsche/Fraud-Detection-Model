# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 20:15:45 2020

@author: Victolo Porsche
"""

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/data.csv')
train, test = train_test_split(data, test_size=0.25, stratify=data['isFraud']) 

train.to_csv("../input/train.csv")
test.to_csv("../input/test.csv")