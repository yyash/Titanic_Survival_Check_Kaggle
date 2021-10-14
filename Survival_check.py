#Titanic Classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
train_dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")

main_dataset = pd.concat([train_dataset,test_dataset],axis=0).reset_index(drop=True)

##Spliiting the dataset into Training set and test set
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#Printing column names
list(main_dataset.columns)

#Removing Unneccessary data
dataset = main_dataset[['Age',
 'Embarked',
 'Fare',
 'Parch',
 'PassengerId',
 'Pclass',
 'Sex',
 'SibSp',
 'Survived']]
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())
dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1

#Y = main_dataset[['Survived']]

#Counting the number of males and females
dataset['Sex'].value_counts()

#Feature Engineering - Has family or not
#dataset['Has_Family'] = dataset.loc[dataset['Family'] > 1 ? 1 : 0]
dataset['Has_Family'] = dataset['Family'] > 1
#dataset['Has_Family'] = dataset.loc[dataset['Family'] > 1] = 1


#Label Encoding Sex & Has_Family
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset['Sex'] = labelencoder.fit_transform(dataset['Sex'])
dataset['Has_Family'] = labelencoder.fit_transform(dataset['Has_Family'])

#OneHotEncoding Pclass and Embarked
#dataset.reshape(1,-1)
#dataset['Pclass'] = oneHotEncoder.fit_transform(dataset['Pclass'])
pClassEncodedDf = pd.get_dummies(dataset.Pclass, prefix="Pclass", drop_first = True)
embarkedEncodedDf = pd.get_dummies(dataset.Embarked, prefix="Emb", drop_first = True)
dataset = pd.concat([dataset,embarkedEncodedDf,pClassEncodedDf],axis=1)

dataset_test = dataset[dataset.Survived.isnull()]
dataset_train = dataset[dataset.Survived.notnull()]

Y = dataset_train[['Survived']]

#Dropping irrelevant columns
dataset_train.drop(['Pclass','Embarked','SibSp','Parch', 'Family','Survived'], axis=1, inplace=True)
dataset_train.drop(['PassengerId'], axis=1, inplace=True)

dataset_test.drop(['Pclass','Embarked','SibSp','Parch', 'Family'], axis=1, inplace=True)


#Spliiting the dataset into Training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset_train,Y,test_size = 0.25,random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
#dataset_test = sc_X.transform(dataset_test)

#Fitting Random Forest Classifier to the training set
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

classifier = RandomForestClassifier(criterion='entropy', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
classifier.fit(X_train, np.ravel(y_train))

#Testing Accuracy
print("RF Accuracy: "+repr(round(classifier.score(X_test, y_test) * 100, 2)) + "%")
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
result_rf=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')
print('The cross validated score for Random forest is:',round(result_rf.mean()*100,2))
y_pred = cross_val_predict(clf,x_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix for RF', y=1.05, size=15)


#Predicting test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Predicting for Test Values given by Kaggle
test_dataset_kaggle = dataset_test.copy()
test_dataset_kaggle.drop(['PassengerId','Survived'], axis=1, inplace=True)
test_dataset_kaggle = sc_X.fit_transform(test_dataset_kaggle)
np.argwhere(np.isnan(test_dataset_kaggle))

test_dataset_kaggle.isnull().sum()


y_pred_kaggle = classifier.predict(test_dataset_kaggle)
y_pred_kaggle_df = pd.DataFrame(y_pred_kaggle)
y_pred_kaggle_df.reset_index(drop=True)
y_pred_kaggle_df.columns = ['Survived']
#a = a.to_frame()
final_submission_a = dataset_test['PassengerId'].to_frame().reset_index(drop=True)
a.reset_index(drop=True)
final_submission = pd.concat([final_submission_a,y_pred_kaggle_df],axis=1)
final_submission.Survived = final_submission.Survived.astype(int)
filename = 'Titanic Predictions.csv'
final_submission.to_csv(filename,index=False)
print('Saved file: ' + filename)



#Visualizing the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = pd.DataFrame(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set.iloc[:, 0].min() - 1, stop = X_set.iloc[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set.iloc[:, 1].min() - 1, stop = X_set.iloc[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set.iloc[np.array(y_set == j).reshape(300,), 0], X_set.iloc[np.array(y_set == j).reshape(300,), 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualizing the test set results
from matplotlib.colors import ListedColormap
X_set, y_set = pd.DataFrame(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set.iloc[:, 0].min() - 1, stop = X_set.iloc[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set.iloc[:, 1].min() - 1, stop = X_set.iloc[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set.iloc[np.array(y_set == j).reshape(100,), 0], X_set.iloc[np.array(y_set == j).reshape(100,), 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()