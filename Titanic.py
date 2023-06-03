import pandas as pd
import numpy as np
data=pd.read_csv("/content/XYZ.csv")
data
data.info()
data.isnull().sum()
data["Age"]=data["Age"].fillna(19.0)

data["Cabin"]=data["Cabin"].fillna(123)
data["Embarked"]=data["Embarked"].fillna("S")
data
data["Embarked"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Name"]=le.fit_transform(data["Name"])
data["Sex"]=le.fit_transform(data["Sex"])
data["Embarked"]=le.fit_transform(data["Embarked"])
data["Ticket"]=le.fit_transform(data["Ticket"])
data
x=data.iloc[:,[2,3,4,5,6,7,8,9]].values
x
y=data.iloc[:,[1]].values
y
data.corr()
from imblearn.over_sampling import SMOTE
s=SMOTE()
x_data,y_data=s.fit_resample(x,y)
from collections import Counter
print(Counter(y_data))
data.hist()
import seaborn as sns
# sns.kdeplot(data["Embarked"])
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_scale=ss.fit_transform(x_data)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scale,y_data,test_size=0.2,random_state=11)
ab1=x_scale.mean()
ab1=round(ab1)
ab1
ab1=x_scale.var()
ab1=round(ab1)
ab1
from sklearn.linear_model import LogisticRegression
l1=LogisticRegression()
l1.fit(x_train,y_train)
y_pred1=l1.predict(x_test)
y_pred1
from sklearn.metrics import accuracy_score
ac1=accuracy_score(y_test,y_pred1)*100
print(ac1)
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=11,metric="minkowski",p=1)
kn.fit(x_train,y_train)
y_pred2=kn.predict(x_test)
y_pred2
from sklearn.metrics import accuracy_score
ac2=accuracy_score(y_test,y_pred2)*100
print(ac2)
from sklearn.svm import SVC
s1=SVC(kernel="linear")
s1.fit(x_train,y_train)
y_pred3=s1.predict(x_test)
y_pred3
from sklearn.metrics import accuracy_score
ac3=accuracy_score(y_test,y_pred3)*100
print(ac3)
from sklearn.naive_bayes import GaussianNB
g1=GaussianNB()
g1.fit(x_train,y_train)
y_pred4=g1.predict(x_test)
y_pred4
from sklearn.metrics import accuracy_score
ac4=accuracy_score(y_test,y_pred4)*100
print(ac4)
from sklearn.ensemble import VotingClassifier
vc=VotingClassifier(estimators=[("LogisticRegression",l1),("SVC",s1),("KNeighborsClassifier",kn),("naive bayes",g1)])
vc.fit(x_train,y_train)
y_pred5=vc.predict(x_test)
y_pred5
from sklearn.metrics import accuracy_score
ac5=accuracy_score(y_test,y_pred5)*100
print(ac5)
from sklearn.ensemble import BaggingClassifier
bg=BaggingClassifier(base_estimator=l1,n_estimators=11,random_state=1)
bg.fit(x_train,y_train)
y_pred6=bg.predict(x_test)
y_pred6
from sklearn.metrics import accuracy_score
ac6=accuracy_score(y_test,y_pred6)*100
print(ac6)
from sklearn.ensemble import RandomForestClassifier
rd=RandomForestClassifier(n_estimators=21,random_state=1)
rd.fit(x_train,y_train)
y_pred7=rd.predict(x_test)
y_pred7
from sklearn.metrics import accuracy_score
ac7=accuracy_score(y_test,y_pred7)*100
print(ac7)
