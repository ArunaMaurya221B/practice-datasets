import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

training_data = pd.read_csv("/home/icts/practice-datasets/Loan_prediction_problem/train.csv")
testing_data = pd.read_csv("/home/icts/practice-datasets/Loan_prediction_problem/test.csv")
sns.set()
testing_data.shape
training_data.shape
print testing_data
#print training_data


#for non-numerical data
#training_data['Property_Area'].value_counts()
#training_data['Education'].value_counts()
training_data.describe()

#filling the NaN values with mode
training_data['Gender'].fillna(training_data['Gender'].mode()[0], inplace=True)
training_data['Married'].fillna(training_data['Married'].mode()[0], inplace=True)
training_data['LoanAmount'].fillna(training_data['LoanAmount'].mode()[0], inplace=True)
training_data['Credit_History'].fillna(training_data['Credit_History'].mode()[0], inplace=True)
training_data['Property_Area'].fillna(training_data['Property_Area'].mode()[0], inplace=True)
training_data['Education'].fillna(training_data['Education'].mode()[0], inplace=True)

testing_data['Gender'].fillna(testing_data['Gender'].mode()[0], inplace=True)
testing_data['Married'].fillna(testing_data['Married'].mode()[0], inplace=True)
testing_data['LoanAmount'].fillna(testing_data['LoanAmount'].mode()[0], inplace=True)
testing_data['Credit_History'].fillna(testing_data['Credit_History'].mode()[0], inplace=True)
testing_data['Property_Area'].fillna(testing_data['Property_Area'].mode()[0], inplace=True)
testing_data['Education'].fillna(testing_data['Education'].mode()[0], inplace=True)

#analysing the relationship
sns.swarmplot(x="Loan_Status", y='ApplicantIncome', hue='Married', palette='Set2', data=training_data)
#plt.show()
plt.close()

sns.swarmplot(x='Loan_Status', y='Credit_History', data=training_data)
#plt.show()
plt.close()

sns.boxplot(x="Loan_Status", y='CoapplicantIncome', hue='Education', palette='Set2', data=training_data)
#plt.show()
plt.close()

sns.boxplot(x="Loan_Status", y='ApplicantIncome', hue='Education', palette='Set2', data=training_data)
#plt.show()
plt.close()

le = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Credit_History', 'ApplicantIncome','CoapplicantIncome']

for i in categorical_columns:
	training_data[i] = le.fit_transform(training_data[i])
	testing_data[i] = le.fit_transform(testing_data[i])

training_data['Loan_Status'] = le.fit_transform(training_data['Loan_Status'])

train_features = training_data.iloc[:, [2, 6, 7, 10]]
train_labels = training_data.iloc[:, [12]]
test_features = testing_data.iloc[:, [2, 6, 7, 10]]
test_labels = testing_data.iloc[:, [11]]

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=20)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_features, train_labels)
pred = clf.predict(test_features)

print clf.score(train_labels, pred)