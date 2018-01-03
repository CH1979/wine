import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic.csv')
X = data
X = X.drop(['PassengerId', 'Name', 'SibSp', 'Ticket', 'Cabin', 'Embarked', 'Parch'], axis=1)
X = X.dropna()
y = X['Survived']
X = X.drop(['Survived'], axis=1)
X = X.replace('male', 1)
X = X.replace('female', 0)
clf = DecisionTreeClassifier(random_state = 241)
clf.fit(X, y)
print(X)
print(clf.feature_importances_)