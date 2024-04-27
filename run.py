import numpy as np

from tree import DecisionTreeClassifier
from bayes import BayesianClassifier
from linear import LogisticRegression
from nn import MLPBinaryClassifier
from base.utils import prepare_dataset, scoring

    
def pipeline(model, train, test):
    name = model.name
    trainX, trainY = train[:, :-1], train[:, -1]
    print("")
    print("Model: %s\n" % name + "-" * 40)
    model.fit(trainX, trainY)
    
    acc, f1, auc = scoring(trainY, model.predict(trainX))
    print("Train Accuracy=%.4f | F1=%.4f | AUC=%.4f" % (acc, f1, auc))

    acc, f1, auc = scoring(test[:, -1].astype(np.int32), model.predict(test[:, :-1]))
    print("Test Accuracy=%.4f | F1=%.4f | AUC=%.4f" % (acc, f1, auc))



if __name__ == "__main__":
    train, test, labels = prepare_dataset("./iris.csv", do_normalize=True)
    for architect in [LogisticRegression, MLPBinaryClassifier]:
        pipeline(architect(), train, test)
    train, test, labels = prepare_dataset("./iris.csv", do_discretize=True)
    for architect in [BayesianClassifier, DecisionTreeClassifier]:
        pipeline(architect(), train, test)
    
        
