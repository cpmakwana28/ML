from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
#calling function
fl_iris=load_iris()

#importing function through library model_selection through which we can split training and testing data
from sklearn.model_selection import train_test_split

#splitting training and testing data with target data also
train_data,test_data,train_target,test_target=train_test_split(fl_iris.data,fl_iris.target,test_size=0.2)

#import numpy as np
#print(np.size(train_data))

#importing library to use algorithm
from sklearn import tree

#calling decesion tree function
algo=tree.DecisionTreeClassifier()

#training machine via decesion tree algo
training=algo.fit(train_data,train_target)

#generating predicted output
predicted_output=training.predict(test_data)

print(predicted_output)

#loading accuracy_score function to check accuracy of output
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(test_target,predicted_output)
print(accuracy)

#importing module to view algorithm
out_data=tree.export_graphviz(algo,
                    out_file=None,
                    feature_names=fl_iris.feature_names,
                    class_names=fl_iris.target_names,
                    filled=True,
                    rounded=True
                    )
graphviz.Source(out_data)