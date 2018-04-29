#***************************** Load iris dataset. Use K Means to cluster all datapoint into cluster using python code. Use elbow method to find number of clusters.*********************************

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

#loading data
datapath="/home/student/Desktop/Theory Sessions/10-3/CSV/iris.csv"
data=pd.read_csv(datapath,sep=",",names=["sepal_length","sepal_width","Petal_length","Petal_width","Class"])
#print(data)

#preprocessing
dic={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
cl=[]
for i in data["Class"]:
    cl.append(dic[i])
data["Class_new"]=pd.Series(cl)

X=data.drop("Class",axis=1)
print(X)
y=data["Class_new"]

#Splitting
x_train=X[:120]
x_test=X[120:]
y_train=y[:120]
y_test=y[120:]
"""
print("x_train=",len(x_train))
print("x_test=",len(x_test))
print("y_train=",len(y_train))
print("y_test=",len(y_test))
"""
distortions = []
K = range(1,10)#possible number of clusters
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

print('distortions=',distortions)
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

k_means=KMeans(n_clusters=3,random_state=0)
k_means.fit(x_train)
predicted=k_means.predict(x_test)
centroids=k_means.cluster_centers_
labels=k_means.labels_

for i,j in zip(pd.Series(predicted),y_test):
    print("Predicted Cluster=",i,'\t',"Actual Cluster=",j)

print('Accuracy Score=', metrics.accuracy_score(predicted,y_test))


"""
#==================================================Output:=======================================================
distortions= [2.0802006916346016, 0.9467986410243767, 0.6671852795977241, 0.6002602672284263, 0.5284154323556666, 0.48413335634239646, 0.44301507282909064, 0.41788673917433583, 0.4007945028681197]
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 0 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Predicted Cluster= 2 	 Actual Cluster= 2
Accuracy Score= 0.9666666666666667
student@student-OptiPlex-3020:~$ 



"""






