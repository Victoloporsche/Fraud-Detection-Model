from data_information import DataInformation
from preprocess_data_strategy import PreprocessDataStrategy
from optimization import GridSearchHelper
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

import pandas as pd


class Main():
    '''
    This class brings all the module together to perform the classification task
    '''

    def __init__(self, train, test):
        print("ObjectOrientedTitanic object created")

        self.testID = test['id']
        self.number_of_train = train.shape[0]

        self.y_train = train['isFraud']
        self.train = train.drop('isFraud', axis=1)
        self.y_test = test['isFraud']
        self.test = test.drop('isFraud', axis=1)


        # concat train and test data
        self.all_data = self._get_all_data()

        # Create instance of objects
        self._info = DataInformation()
        self.preprocessStrategy = PreprocessDataStrategy()
        self.gridSearchHelper = GridSearchHelper()

    def _get_all_data(self):
        return pd.concat([self.train, self.test])


    def mssing_values(self):
        """
        using _info object gives summary about dataset
        :return:
        """
        self._info.number_missing(self.all_data)

    def discrete_outliers(self):
        self._info.discrete_detect_outliers(self.all_data)

    def continous_outliers(self):
        self._info.continous_detect_outliers(self.all_data)

    def preprocessing(self, strategy_type):
        """
        Process data depend upon strategy type
        :param strategy_type: Preprocessing strategy type
        :return:
        """
        self.strategy_type = strategy_type

        self.all_data = self.preprocessStrategy.strategy(self._get_all_data(), strategy_type)


    def machine_learning(self):
        """
        Get self.X_train, self.X_test and self.y_train
        Find best parameters for classifiers registered in gridSearchHelper
        :return:
        """
        self._get_train_and_test()

        self.gridSearchHelper.fit_predict_save(self.X_train,
                                               self.X_test,
                                               self.y_train,
                                               self.y_test,
                                               self.testID,
                                               self.strategy_type)

    def show_cross_validation_result(self):
        self.gridSearchHelper.show_result()

    def _get_train_and_test(self):
        """
        Split data into train and test datasets
        :return:
        """
        self.X_train = self.all_data[:self.number_of_train]
        self.X_test = self.all_data[self.number_of_train:]

    def feature_selection(self, train, strategy_type):
        train = self.preprocessStrategy.strategy(train, strategy_type)
        ytrain = train[['isFraud']]
        xtrain = train.drop(['id', 'isFraud'], axis=1)
        feature_sel_model = ExtraTreesClassifier().fit(xtrain, ytrain)
        ranked_features = pd.Series(feature_sel_model.feature_importances_, index= xtrain.columns)
        ranked_features.nlargest(10).plot(kind='bar')
        plt.show()



