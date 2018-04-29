#************************************Load breast cancer CSV. Treat string column bare nuclei and convert it into integers by iterating over pandas series and typecasting it in integer. Design random forest classifier to predict type of the cancer.********************************


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#loading data
datapath="/home/student/Desktop/Theory Sessions/11-3/CSV/breast_cancer.csv"
data=pd.read_csv(datapath,sep=",",names=["Sample code number","Clump thickness","Uniformity of cell size","Uniformity of cell shape","Marginal adhesion","Single epithelial cell size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","class"])

#splitting dataset
X=data.drop(["Bare Nuclei","class"],axis=1)
y=data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=123,stratify=y) 

#Training data
clf=RandomForestClassifier(n_estimators=700,max_features='log2') 
clf.fit(X_train, y_train)
importances = clf.feature_importances_
indices = np.argsort(importances)
pred = clf.predict(X_test)

#Printing output
for i,j in zip(pd.Series(pred),y_test):
    print("Predicted=",int(round(i)),"\tActual=",j)

print('Accuracy Score=', metrics.accuracy_score(pred,y_test))



#***********************************************Output********************************************
"""
Predicted= 2 	Actual= 2
Predicted= 4 	Actual= 4
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2
Predicted= 4 	Actual= 4
Predicted= 2 	Actual= 2
Predicted= 4 	Actual= 2
Predicted= 4 	Actual= 4
Predicted= 2 	Actual= 2
Predicted= 4 	Actual= 4
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2
Predicted= 4 	Actual= 2
Predicted= 4 	Actual= 4
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2
Predicted= 4 	Actual= 4
Predicted= 2 	Actual= 2
Predicted= 4 	Actual= 4
Predicted= 2 	Actual= 2
Predicted= 4 	Actual= 4
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2
Predicted= 4 	Actual= 4
Predicted= 4 	Actual= 4
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2
Predicted= 4 	Actual= 4
Predicted= 2 	Actual= 2
Predicted= 4 	Actual= 4
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2
Predicted= 2 	Actual= 2


Accuracy Score= 0.9571428571428572

"""
