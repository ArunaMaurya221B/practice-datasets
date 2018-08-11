import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import string
import collections
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

data_set = pd.read_csv("Iris.csv")

#print data_set
#data_set = data_set.sample(frac=1).reset_index(drop=True)

sns.set()

plt.hist(data_set['SepalLengthCm'])
plt.xlabel('Sepal length')
plt.ylabel('Count')
plt.show()
plt.close()

plt.hist(data_set['SepalWidthCm'])
plt.xlabel('Sepal width')
plt.ylabel('Count')
plt.show()
plt.close()

plt.hist(data_set['PetalLengthCm'])
plt.xlabel('Petal length')
plt.ylabel('Count')
plt.show()
plt.close()

plt.hist(data_set['PetalWidthCm'])
plt.xlabel('Petal width')
plt.ylabel('Count')
plt.show()
plt.close()

sns.swarmplot(x='Species', y='PetalLengthCm', data=data_set)
name="img.png"
plt.savefig("name.png")
plt.show()
plt.close()

data_set = data_set.as_matrix()
X = data_set[:, 1:5]
y=data_set[:, 5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print knn.score(X_test, y_test)
print knn.score(X_train, y_train)