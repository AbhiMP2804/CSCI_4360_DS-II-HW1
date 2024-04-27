import torch as tc


from base.module import GradientClassifier


"""
TODO

You will implement a linear model in this file.

"""


class LogisticRegression(GradientClassifier):
    def __init__(self, feat_dim=4):
        GradientClassifier.__init__(self, "LogReg")
        self._features = feat_dim
        self.weights = tc.nn.Parameter(tc.randn((feat_dim, 1), dtype=tc.float32))
        self.bias = tc.nn.Parameter(tc.zeros((1, ), dtype=tc.float32) + 0.1)

    def forward(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self._features
        "Q1. Fill the line below. Linear transformation with an offset parameter."
        logits = tc.matmul(x, self.weights) + self.bias  #-------------------------------------------------------------------->>>>>>>>>>> COMPLETED
        return tc.sigmoid(logits)
