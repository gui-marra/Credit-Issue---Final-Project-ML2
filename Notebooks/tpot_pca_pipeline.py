import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.1830263283386442
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.8500000000000001, min_samples_leaf=19, min_samples_split=17, n_estimators=100)),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=1, min_samples_leaf=10, min_samples_split=4)),
    GaussianNB()
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
