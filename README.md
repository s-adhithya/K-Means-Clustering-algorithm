# Implementation of K-Means Clustering Algorithm
## Aim
To write a python program to implement K-Means Clustering Algorithm.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation

## Algorithm:
## Step 1:
Import pandas.
## Step 2:
Import matplotlib.pyplot.
## Step 3:
Import sklearn.cluster from KMeans module.
## Step 4:
Import seaborn.
## Step 5:
Import warnings
## Step 6:
Declare warnings.filerwarning with ignore as argument
## Step 7:
Declare a variable x1 and read a csv file(clustering.csv) in it.
## Step 8:
Declare a variable x2 as index of x1 with arguments ApplicantIncome and LoanAmount.
## Step 9:
Display x1.head(2) and x2.head(2).
## Step 10:
Declare a variable x and store x2.values.
## Step 11:
Declare sns.scatterplot for ApplicantIncome and LoanAmount by indexing.
## Step 12:
Plot Income , Loan and display them.
## Step 13:
Declare a variable kmean = KMean(n_cluster_centers_) and execute kmean.fit(x).
## Step 14:
Display kmean.cluster)centers
## Step 15:
Display kmean.labels_ 
## Step 16:
Declare a variable predcited_class to kmean.predict([[]]) and give two arguments in it.
## Step 17:
Display the predicted_class

## Program:
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('clustering(1).csv')
print(data.head(2))

x1=data.loc[:,['ApplicantIncome','LoanAmount']]
print(x1.head(2))

X=x1.values
sns.scatterplot(X[:,0],X[:,1])
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()

Kmean=KMeans(n_clusters=4)
Kmean.fit(X)

print('Cluster Centers: ',Kmean.cluster_centers_)
print('Labels:',Kmean.labels_)

predicted_cluster=Kmean.predict([[9200,110]])
print('The clusters group for the Application 9200 and Loan Amount 110  is ',predicted_cluster)
```
## Output:
![output](/filename16.png)
## Result
Thus the K-means clustering algorithm is implemented and predicted the cluster class using python program.