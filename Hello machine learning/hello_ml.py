
"""
This is python file contain codes for basic hello machine learning for begginers buy using the iris data set to start ML and also understand the basic machine learning
"""
""" importing the libraries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


"""loading the dataset using pandas"""
"""path to the dataset"""
path="../datasets/iris.csv" 
dataset=pd.read_csv(path)

"""
use shape function to see the numbber of rows and colunms
(row,columns)
""" 
#print(dataset.shape) 

"""
use describe function to get the summury of the dataset
"""
# print(dataset.describe())

"""use the keys() function to see the 'titles' of the dataset
"""
# print(dataset.keys())

"""use 'variety'-columns as in the dataset- to to see the species  that we want to predict

Used loop to avoid repetation of the data available
"""
# species=[]
# for i in dataset['variety']:
#     if i in species:
#         pass
#     else:
#         species.append(i)
# print(species)

"""
Use fillna to fill any row that us left blank
#data cleaning
"""
# print(dataset.fillna(0))
"""
visualizing your data to check for anomalities
"""


sns.pairplot(dataset,hue="variety")
# plt.show()
"""
==================SPLIT THE DATA===============
"""
"""
define the dependant and indipendent values
"""
dataset=dataset.values
x=dataset[:,0:4]
y=dataset[:,4]

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0)


    
"""
TRAIN THE MODEL-K-NEAREST NEIGHBOR
"""
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
 
"""
TESTING THE MODEL USING THE REST 25% OF THE DATA
"""
# print(knn.predict(X_test))

"""
EVALUATING THE MODEL ACCURACY
"""

y_predict=knn.predict(X_test)
# print(f"The model accuracy is {np.mean(y_predict==y_test)*100}%")




































