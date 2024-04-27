import numpy as np

from base.module import StatisticClassifier

"""
TODO

You will implement a Bayesian Classifier in this file.

Q1:
    During the inference, if we have calculate the posterior probability
    P(Y=y|X), how we can find out the target label with highest P(Y=y|X)?

Q2:
    During the training, we need to store all possible probability [P(X1|Y=y), P(X2|Y=y), ...., P(Xi|Y=y)]
    Here, Xi indicates the i-th variable. Further more, each probability P(Xi|Y=y) should obtain
    all possible values of Xi. That is we have P(Xi|Y=y) = [P(Xi1|Y=y), P(Xi2|Y=y), ..., P(Xij|Y=y)].
    Implement codes to store P(Xi|Y=y).
    
"""


class BayesianClassifier(StatisticClassifier):
    
    def __init__(self):
        StatisticClassifier.__init__(self, "NaiveBayesian")
        self._probas = None

    def forward(self, sample):
        target_y, target_prob = 0, 0.0
        for y, (postior, cond_probs) in enumerate(self._probas):
            # P(Y=y|x1, x2, ...) ~= P(Y=y) * P(X1=x1|Y=y) * P(X2=x2|Y=y) * ...
            for x_idx, x_cond_prob_dict in enumerate(cond_probs):
                postior *= x_cond_prob_dict.get(sample[x_idx], 0.0)
            "Q1. Fill the line below. Find the highest posterior probability."
            if postior > target_prob: #-------------------------------------------------------------------->>>>>>>>>>> COMPLETED
                target_y, target_prob = y, postior
        return target_y            

    def _fit(self, X, Y):
        # the dimension of is X = (num_samples, num_features)
        self._probas = [] # store P(y) and P(X|Y=y) as a tuple
        for y in range(self._num_cls):
            # calculate P(X|Y=y) by storing them as a sequence
            subX = X[Y == y] # records with label y
            y_cond_prob = [] # store [P(X1|Y=y), P(X2|Y=y), ...] as list
            for subx_seq in subX.T:
                counts = {}
                for val in subx_seq.tolist():
                    if val not in counts:
                        counts[val] = 0
                    counts[val] += 1
                "Q2. Fill the line below. Calculate P(X1=x11|Y=y), P(X1=x21|Y=y), hint: use dict() data type."
                x_probs = {value: counts[value] / len(subx_seq) for value in counts} #----------------------------------->>>>>>>>>>> COMPLETED 
                y_cond_prob.append(x_probs) # the dimension of y_cond_prob is (2,4) in Iris dataset.
                
            Py = len(subX) / len(X) # calculate P(Y=y)
            self._probas.append((Py, y_cond_prob))
