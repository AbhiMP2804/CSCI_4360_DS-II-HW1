import torch as tc


from base.module import GradientClassifier


"""
TODO

You will implement a 1-hidden-layer (i.e., 2-layer) MLP in this file.

"""


class MLPBinaryClassifier(GradientClassifier):
    def __init__(self, in_dim=4, hide_dim=64, device="cpu"):
        GradientClassifier.__init__(self, "MLP")
        self._features = in_dim
        self.weights_hide = tc.nn.Parameter(tc.randn((in_dim, hide_dim)))
        self.bias_hide = tc.nn.Parameter(tc.zeros((hide_dim, ), dtype=tc.float32) + 0.1)
        self.weights_clf = tc.nn.Parameter(tc.randn((hide_dim, 1)))
        self.bias_clf = tc.nn.Parameter(tc.zeros((1, ), dtype=tc.float32) + 0.1)

    def forward(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self._features
        # Linear transformation and bias for the hidden layer
        hidden_output = tc.matmul(x, self.weights_hide) + self.bias_hide #-------------------------------------------------------------------->>>>>>>>>>> COMPLETED
        # Nonlinear activation function (you can use ReLU or any other activation)
        hidden_output = tc.relu(hidden_output) #-------------------------------------------------------------------->>>>>>>>>>> COMPLETED
        # Linear transformation and bias for the output layer
        logits = tc.matmul(hidden_output, self.weights_clf) + self.bias_clf #-------------------------------------------------------------------->>>>>>>>>>> COMPLETED
        
        """Q1. Implement the MLP model (~3 lines of code would work). 
        Linear transformation -> nonlinear mapping -> Linear transformation"""
        
        return tc.sigmoid(logits)
