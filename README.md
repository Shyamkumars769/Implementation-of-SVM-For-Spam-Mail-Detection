# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the Program.

## Program:

Program to implement the SVM For Spam Mail Detection..

Developed by: Shyam Kumar.S

RegisterNumber:  212224040315

```
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()
data.info()
data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

![image](https://github.com/user-attachments/assets/04a882fa-8c1e-44b4-88be-32e44464fc69)

![image](https://github.com/user-attachments/assets/ef247c3f-3bb8-4374-85ba-dd416fa1bc08)

![image](https://github.com/user-attachments/assets/9645eff3-4682-4e37-9330-27ca390b4bbb)

![image](https://github.com/user-attachments/assets/a4a9098a-2f47-4f00-8f55-ebbdc9dba040)

![image](https://github.com/user-attachments/assets/e9e7a8bb-bc02-403b-9ec7-57a1ef581583)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
