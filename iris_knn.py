from sklearn.datasets import load_iris
#load iris datasets in a variable
fl_iris=load_iris()

from sklearn.model_selection import train_test_split
#splitting data
train_data,test_data,train_target,test_target=train_test_split(fl_iris.data,fl_iris.target,test_size=0.2)

#importing module to use K-NN algorithm
from sklearn import neighbors

#calling K-NN algorithm
algo=neighbors.KNeighborsClassifier(n_neighbors=3)
#training algo
training=algo.fit(train_data,train_target)
#tresting algo
predicted_output=training.predict(test_data)

print(predicted_output)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_target,predicted_output))