import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ensure reproducibility
torch.manual_seed(0)

# Create a toy dataset
X_pos = torch.rand(100, 4)
X_neg = torch.rand(100, 4) - 0.8
X = torch.cat((X_pos, X_neg), 0)
y = torch.cat((torch.ones(100), torch.zeros(100))) # 100 positive and 100 negative samples

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Define the logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        # Define the linear layer
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # Model definition (complete this part)
	
        return y_pred

# Model initialization
input_size = 4
model = LogisticRegressionModel(input_size)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Testing metric
def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        outputs = outputs.squeeze()
        predicted = outputs.round()
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        return accuracy

# Training loop
def train_model(model, criterion, optimizer, X_train, y_train, epochs=100):
    model.train()
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        outputs = outputs.squeeze()
        loss = criterion(outputs, y_train)

        # Backward pass and optimization (please complete this part)
	

        # monitor performance during training
        if epoch % 10 == 0 or epoch == epochs - 1:
          accuracy = test_model(model, X_test, y_test)
          print(accuracy)

# Train the model
train_model(model, criterion, optimizer, X_train, y_train)

# Calculate accuracy
accuracy = test_model(model, X_test, y_test)
print("Final accuracy:", accuracy)

