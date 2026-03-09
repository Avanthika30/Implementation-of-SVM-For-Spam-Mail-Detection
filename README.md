# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.  

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: AVANTHIKA.B
RegisterNumber: 212224040039 
*/
```
```
import pandas as pd
data=pd.read_csv("C:\\Users\\admin\\Downloads\\spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
print("Accuracy:",acc,end='\n\n')

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```
## Output:
### Data:
<img width="1048" height="657" alt="image" src="https://github.com/user-attachments/assets/d5f46f59-fe4f-46d6-9226-e71925f0e03c" />

### Confusion Matrix:
<img width="130" height="63" alt="image" src="https://github.com/user-attachments/assets/535b4da0-f107-412f-be18-735aa9d6a5f7" />

### Accuracy:
<img width="300" height="40" alt="image" src="https://github.com/user-attachments/assets/b00ce873-5f56-4c40-8019-6a7442b84af4" />

### Classification Report:
<img width="547" height="170" alt="image" src="https://github.com/user-attachments/assets/cbccce30-6d2e-4deb-b3fd-dae9d720669f" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
