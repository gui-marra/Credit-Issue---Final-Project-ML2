# util
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##############################################################################################

##Classifiers
#Supervised
from sklearn.ensemble import AdaBoostClassifier #begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted
from sklearn.ensemble import BaggingClassifier #Bagging classifier fits base classifiers each on random subsets of the original dataset and aggregate their individual predictions
from sklearn.ensemble import ExtraTreesClassifier #Extremely Random Trees: This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
from sklearn.ensemble import GradientBoostingClassifier #GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions
from sklearn.ensemble import RandomForestClassifier # random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import RidgeClassifier #Classifier using Ridge regression. This classifier first converts the target values into {-1,1} and then treats the problem as a regression task
from sklearn.linear_model import LogisticRegression #Logistic Regression (aka logit, MaxEnt) classifier.
from sklearn.linear_model import LogisticRegressionCV #Logistic Regression CV (aka logit, MaxEnt) classifier.
from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier #Classifier implementing the k-nearest neighbors vote.
from sklearn.neighbors import RadiusNeighborsClassifier #Classifier implementing a vote among neighbors within a given radius

from sklearn.svm import SVC #Support Vector Classification
 
from pygam import LogisticGAM, s, f #interesting stuff

from sklearn.ensemble import VotingClassifier #Voting for the best classifiers

def load_dataset(name):
    print(f"Loading dataset: {name}")
    
    df = pd.read_csv(name)
    variable_names = list(df)
    aux = []
    for elem in variable_names:
        if elem != "Y":
            aux.append(elem)
            
    print(f"Variable Names: {variable_names}")
    
    df.fillna(0, inplace=True)
    
    X, y = df[aux], df["Y"]
    
    return X, y

def model():
    clf = [
        [AdaBoostClassifier(), "AdaBoostClassifier"],
        [BaggingClassifier(), "BaggingClassifier"],
        [ExtraTreesClassifier(), "ExtraTreesClassifier"],
        [GradientBoostingClassifier(), "GradientBoostClassifier"],
        [RandomForestClassifier(), "RandomForestClassifier"],
        
        [DecisionTreeClassifier(), "DecisionTreeClassifier"],
        
        #[GaussianNB(), "GaussianNB"],
        
        [RidgeClassifier(), "RidgeClassifier"],
        [LogisticRegression(), "LogisticRegression"],
        [LogisticRegressionCV(), "LogisticRegressionCV"],
        [SGDClassifier(), "SGDClassifier"],
        [KNeighborsClassifier(), "KNeighborsClassifier"]
        #[RadiusNeighborsClassifier(), "RadiusNeighborsClassifier"]
        #[SVC(), "SVC"]
    ]
    
    performance_train = {}
    performance_test = {}
    filename = "../Notebooks/data1.csv"
    X, y = load_dataset(filename)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    
    for classifier, clf_name in clf: performance_train[clf_name] = []
    for classifier, clf_name in clf: performance_test[clf_name] = []
    
    for elem in clf: #fit the models above using default hyperparameters
        classifier = elem[0]
        classifier_name = elem[1]
        print(classifier_name)
        try:    
            classifier.fit(X_train, y_train)
            
            y_hat = classifier.predict(X_train)
            
            f1_train = f1_score(y_train, y_hat)
            accuracy_train = accuracy_score(y_train, y_hat)
            precision_train = precision_score(y_train, y_hat)
            recall_train = recall_score(y_train, y_hat)
            print(f"Train scores: \tf1-score: {f1_train}\tAccuracy: {accuracy_train}\tPrecision: {precision_train}\tRecall: {recall_train}")
            performance_train[classifier_name].append(f1_train)
            performance_train[classifier_name].append(accuracy_train)
            performance_train[classifier_name].append(precision_train)
            performance_train[classifier_name].append(recall_train)
            
            y_pred = classifier.predict(X_test)
            
            f1_test = f1_score(y_test, y_pred)
            accuracy_test = accuracy_score(y_test, y_pred)
            precision_test = precision_score(y_test, y_pred)
            recall_test = recall_score(y_test, y_pred)            
            print(f"Test scores: \tf1-score: {f1_test}\tAccuracy: {accuracy_test}\tPrecision: {precision_test}\tRecall: {recall_test}")
            performance_test[classifier_name].append(f1_test)
            performance_test[classifier_name].append(accuracy_test)
            performance_test[classifier_name].append(precision_test)
            performance_test[classifier_name].append(recall_test)

            print("\n**********************************************************************")
        except ImportError:
            print("Classifier \"" + classifier_name + "failed.")
              
    try:    #Plot the results above for train and test datasets
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_style("whitegrid")
                        
        classifiers = []
        f1 = []
        acc = []
        prc = []
        rcl = []
            
        for key in performance_train:
                classifiers.append(key)
                classifiers.append(key)
                
                f1.append(performance_train[key][0])
                f1.append(performance_test[key][0])
                
                acc.append(performance_train[key][1])
                acc.append(performance_test[key][1])
                
                prc.append(performance_train[key][2])
                prc.append(performance_test[key][2])
                
                rcl.append(performance_train[key][3])
                rcl.append(performance_test[key][3])
                
        results = {"Classifier": classifiers, "F1": f1, "Accuracy": acc, "Precision": prc, "Recall": rcl}
        df_results = pd.DataFrame(results) 
        df_results.to_csv("results_trial1.csv")
        #for key in results:
         #   plt.figure()
          #  sns_plot = sns.barplot(x = 'Classifier', y = key, data = df_results)
          #  sns_plot.set_xlabel('Classifier', fontsize = 15)
          #  sns_plot.set_ylabel(key, fontsize = 15)
          #  sns_plot.tick_params(labelsize = 15)
            #Separate legend from graph
          #  plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., fontsize = 15)
          #  string = key + ".png"
          #  print(string)
           # plt.savefig(string)
           # string = ''
           # plt.close()
        
    except ImportError:
        print("Cannot import matplotlib or seaborn.")
            
    #write the summary in summary.txt
    f = open("summary.txt", "w")
    
    f.write("Train results:\n")
    for classifier in performance_train:
        f.write(f"{classifier} -> F1:{performance_train[classifier][0]} \tAccuracy: {performance_train[classifier][1]}\tPrecision: {performance_train[classifier][2]}\tRecall: {performance_train[classifier][3]} \n")    
            
    f.write("Test results:\n")
    for classifier in performance_test:
        f.write(f"{classifier} -> F1:{performance_test[classifier][0]} \tAccuracy: {performance_test[classifier][1]}\tPrecision: {performance_test[classifier][2]}\tRecall: {performance_test[classifier][3]} \n")    
    
    f.close()       
    
    
    pass


if __name__ == "__main__":
    model()
    pass