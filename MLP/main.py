from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load iris dataset
dataset=pd.read_csv("../datasets/iris.csv")

dataset=dataset.values
x=dataset[:,0:4]
y=dataset[:,4]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Create MLP classifier object with parameters
mlp = MLPClassifier(hidden_layer_sizes=(100,),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    power_t=0.5,
                    max_iter=200,
                    shuffle=True,
                    random_state=None,
                    tol=0.0001,
                    verbose=True,
                    warm_start=False,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    early_stopping=False,
                    validation_fraction=0.1,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-08,
                    n_iter_no_change=10)

# Train the model using the data
mlp.fit(X_train, y_train)

# Predict target for test data
y_pred = mlp.predict(X_test)
print("Predicted target:", y_pred)

print(f"The model accuracy is {np.mean(y_pred==y_test)*100}%")
""""
Explanation of parameters:
    hidden_layer_sizes: the number of neurons in each hidden layer.
    activation: the activation function to be used in the hidden layer.
    solver: the algorithm used to optimize the weights.
    alpha: the regularization parameter.
    batch_size: the size of the mini-batches.
    learning_rate: the learning rate schedule for weight updates.
    learning_rate_init: the initial learning rate.
    power_t: the exponent for inverse scaling of learning rate.
    max_iter: the maximum number of iterations.
    shuffle: whether to shuffle the training data at each iteration.
    random_state: the seed used by the random number generator.
    tol: the tolerance for stopping criterion.
    verbose: whether to print progress messages.
    warm_start: whether to reuse the solution of the previous call to fit as initialization.
    momentum: the momentum for gradient descent.
    nesterovs_momentum: whether to use Nesterov's momentum.
    early_stopping: whether to use early stopping to terminate training when validation score is not improving.
    validation_fraction: the proportion of the training data to use as validation data.
    beta_1: the exponential decay rate



"""