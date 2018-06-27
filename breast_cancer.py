#import breast cancer dataset
from sklearn.datasets import load_breast_cancer
#store breast acncer dataset in a variable
breast_can=load_breast_cancer()
#import module for splitting data
from sklearn.model_selection import train_test_split
#spliting data for testing and training
train_data,test_data,train_target,test_target=train_test_split(breast_can.data,breast_can.target,test_size=0.7)
#import tree module
from sklearn import tree
#calling decision tree
algo=tree.DecisionTreeClassifier()
#trainning data
training=algo.fit(train_data,train_target)
#testing data
tested_data=training.predict(test_data)

print(tested_data)

from sklearn.metrics import accuracy_score
#check accuracy data
accuracy=accuracy_score(test_target,tested_data)
print(accuracy*100)

print(breast_can.target_names)

import graphviz
out_data=tree.export_graphviz(algo,
                    out_file=None,
                    feature_names=breast_can.feature_names,
                    class_names=breast_can.target_names,
                    filled=True,
                    rounded=True
                    )
graphviz.Source(out_data)