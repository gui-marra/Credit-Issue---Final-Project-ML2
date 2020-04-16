#Here we will implement the code to find the better parameters to the best Classifiers found
#in model.py 
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV

##################################################################################################

##Classifiers
from sklearn.ensemble import AdaBoostClassifier #begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted
from sklearn.ensemble import BaggingClassifier #Bagging classifier fits base classifiers each on random subsets of the original dataset and aggregate their individual predictions
from sklearn.ensemble import ExtraTreesClassifier #Extremely Random Trees: This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
from sklearn.ensemble import GradientBoostingClassifier #GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier #Classifier implementing the k-nearest neighbors vote.

from sklearn.ensemble import VotingClassifier
