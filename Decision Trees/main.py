import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dataset=pd.read_csv("../datasets/iris.csv")

dataset=dataset.values
x=dataset[:,0:4]
y=dataset[:,4]

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0)

dct=DecisionTreeClassifier(criterion='gini',
                                   splitter='best',
                                   max_depth=None,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0,
                                   max_features=None,
                                   random_state=None,
                                   max_leaf_nodes=None,
                                   min_impurity_decrease=0.0,
                                   class_weight=None,
                                   )
dct.fit(X_train,y_train)

y_predict=dct.predict(X_test)

print(f"The model accuracy is {np.mean(y_predict==y_test)*100}%")

"""
    criterion: the function to measure the quality of a split. It can be "gini" for the Gini impurity or "entropy" for the information gain.
    splitter: the strategy used to choose the split at each node. It can be "best" to choose the best split, or "random" to choose the best random split.
    max_depth: the maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.
 
    min_samples_leaf: the minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf: the minimum weighted fraction of the input samples required to be at a leaf node.
    max_features: the number of features to consider when looking for the best split.
    random_state: the seed used by the random number generator.
    max_leaf_nodes: the maximum number of leaf nodes.
    min_impurity_decrease: the minimum decrease in impurity required to split the node.
    min_impurity_split: threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
    class_weight: weoghts associated with classes in the form {class_label: weight}.
 


"""