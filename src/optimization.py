import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import pickle



class GridSearchHelper():
    def __init__(self):
        print("RandomizedSearchCV Created")

        self.RandomizedSearchCV = None
        self.clf_and_params = list()

        self._initialize_clf_and_params()

    def _initialize_clf_and_params(self):

        clf = DecisionTreeClassifier()
        params = {'max_features': ['auto', 'sqrt', 'log2'],
                  'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                  'min_samples_leaf': [1],
                  'random_state': [123]}
        self.clf_and_params.append((clf, params))

        clf = RandomForestClassifier()
        params = {'n_estimators': [4, 6, 9],
                  'max_features': ['log2', 'sqrt', 'auto'],
                  'criterion': ['entropy', 'gini'],
                  'max_depth': [2, 3, 5, 10],
                  'min_samples_split': [2, 3, 5],
                  'min_samples_leaf': [1, 5, 8]
                  }
        self.clf_and_params.append((clf, params))

        clf = xgb.XGBClassifier()
        params = {'max_depth': [5, 6, 7],
                  'learning_rate' : [0.01,0.05,0.1,0.3,1],
                  'n_estimators' : [50,100,150,200]
                  }
        self.clf_and_params.append((clf, params))

    def fit_predict_save(self, X_train, X_test, y_train, y_test, test_id, strategy_type):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.test_id = test_id
        self.strategy_type = strategy_type

        clf_and_params = self.get_clf_and_params()
        models = []

        self.results = {}

        for clf, params in clf_and_params:
            self.current_clf_name = clf.__class__.__name__
            random_search_clf = RandomizedSearchCV(clf, params, cv=5)
            random_search_clf.fit(self.X_train, self.y_train)
            self.Y_pred = random_search_clf.predict(self.X_test)
            auc_test = roc_auc_score(y_test, self.Y_pred)
            clf_train_acc = round(random_search_clf.score(self.X_train, self.y_train) * 100, 2)
            print(self.current_clf_name, " trained and used for prediction on test data...")
            print('the auc on the test set is:', auc_test)
            self.results[self.current_clf_name] = clf_train_acc

            models.append(clf)

            self.save_result()
            print()

    def show_result(self):
        for clf_name, train_acc in self.results.items():
            print("{} cross validation accuracy is {:.3f}".format(clf_name, train_acc))



    def save_result(self):
        Submission = pd.DataFrame({'id': self.test_id,
                                   'isFraud': self.Y_pred})
        file_name = "{}_{}.csv".format(self.strategy_type, self.current_clf_name.lower())
        Submission.to_csv(file_name, index=False)

        print("Submission saved file name: ", file_name)

    def get_clf_and_params(self):
        return self.clf_and_params

    def add(self, clf, params):
        self.clf_and_params.append((clf, params))








