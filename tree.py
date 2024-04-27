import numpy as np

from base.module import StatisticClassifier



"""
TODO

You will implement the most crucial parts of a Decision Tree in this file.

Q1: calculate the entropy.

Q2: calculate the conditional entropy.

Q3 & Q4: conditions to stop the tree construction.

Q5: recursively construct the sub-trees.

"""


class _Tree:
    
    """basic node structure for construction tree"""
    
    def __init__(self, label, feature=None):
        """Both `label` and `feature` are integers,
           `feature` is the feature index for the next splitting,
           `label` is the predict label at this node
        """
        assert (isinstance(feature, int) and 0 <= feature) or feature is None
        assert isinstance(label, (int, np.int32, np.int64)) and 0 <= label
        self._feature = feature
        self._label = label
        self._children = {}

    @property
    def label(self):
        return self._label

    @property
    def feature(self):
        return self._feature
        
    def __getitem__(self, condition):
        return self._children.get(condition, self._label)

    def __setitem__(self, condition, children):
        assert not self.is_leaf(), 'current node is a leaf!'
        assert isinstance(children, _Tree)
        self._children[condition] = children

    def is_leaf(self):
        return self._feature is None


def standard_entropy(seq):
    assert len(seq.shape) == 1
    uniqs, counts = np.unique(seq, return_counts=True)
    probs = counts / seq.size
    "Q1. Fill the line below. Hint: entropy = -sum[P(y) * log(P(y))]"
    entropy =-np.sum(probs * np.log2(probs)) #-------------------------------------------------------------------->>>>>>>>>>> COMPLETED
    return entropy


def conditional_entropy(x_seq, y_seq):
    assert len(x_seq.shape) == len(y_seq.shape) == 1
    assert x_seq.size == y_seq.size
    entropy = 0.0
    x_uniqs, x_counts = np.unique(x_seq, return_counts=True)
    for x_label, x_count in zip(x_uniqs, x_counts):
        x_prob = x_count / x_seq.size
        y_controled_by_x = y_seq[x_seq == x_label]
        y_entropy_given_x = standard_entropy(y_controled_by_x)
        "Q2. Fill the line below. Hint: conditional entropy = - Px * sum[P(y|x) * logP(y|x)]"
        entropy += x_prob * y_entropy_given_x #-------------------------------------------------------------------->>>>>>>>>>> COMPLETED
    return - entropy
        

class DecisionTreeClassifier(StatisticClassifier):
    def __init__(self):
        StatisticClassifier.__init__(self, "DecisionTree")
        self._tree = None

    def forward(self, sample):
        tree = self._tree
        while not tree.is_leaf():
            tree = tree[sample[tree.feature]]
        return tree.label

    def _construct_tree(self, X, Y, used_fids):
        #We generate the decision tree recursively. Specifically, there
        #are three steps as following:
        #Step-1: check whether we stop the recursion.
        #Step-2: calculate information gain of each variable to pick up
        #        the best variable to split data.
        #Step-3: recursively call this function to generate children
        #        by using the best splitting variable.

        # Step-1: check whether we need to generate children
        most_freq_y = np.bincount(Y.astype(np.int32)).argmax()
        entire_entropy = standard_entropy(Y)
        "Q3. Fill the line below. Hint: assign a bool value to represent: whether there is only a single class."
        stop_cond_1 = len(np.unique(Y)) == 1 #-------------------------------------------------------------------->>>>>>>>>>> COMPLETED
        "Q4. Fill the line below. Hint: assign a bool value to represent: whether we have used all features."
        stop_cond_2 = len(used_fids) == X.shape[1] #-------------------------------------------------------------------->>>>>>>>>>> COMPLETED
        if stop_cond_1 or stop_cond_2:
            return _Tree(most_freq_y) 
        
        # Step-2: find out the best splitting feature
        max_info_gain, best_fid, best_seq = -float("inf"), 0, None
        for fid, x_seq in enumerate(X.T):
            if fid in used_fids:
                continue
            x_info_gain = entire_entropy - conditional_entropy(x_seq, Y)
            if x_info_gain > max_info_gain:
                max_info_gain, best_fid, best_seq = x_info_gain, fid, x_seq

        # Step-3: recursively generate children of the current tree
        root = _Tree(most_freq_y, best_fid)
        used_fids = used_fids | {best_fid}
        for uniq_val in np.unique(best_seq):
            uniq_idx = best_seq == uniq_val
            subX, subY = X[uniq_idx], Y[uniq_idx]
            "Q5. Fill the line below. Hint: recursively call `self._construct_tree` to generate children."
            root[uniq_val] =self._construct_tree(subX, subY, used_fids) #-------------------------------------------------------------------->>>>>>>>>>> COMPLETED
        return root

    def _fit(self, X, Y):
        self._tree = self._construct_tree(X, Y, set())
