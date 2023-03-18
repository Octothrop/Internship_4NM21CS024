import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN

df = pd.read_csv('CC GENERAL.csv')
print(df.describe())
"""Output:
            BALANCE  BALANCE_FREQUENCY  ...  PRC_FULL_PAYMENT       TENURE
count   8950.000000        8950.000000  ...       8950.000000  8950.000000
mean    1564.474828           0.877271  ...          0.153715    11.517318
std     2081.531879           0.236904  ...          0.292499     1.338331
min        0.000000           0.000000  ...          0.000000     6.000000
25%      128.281915           0.888889  ...          0.000000    12.000000
50%      873.385231           1.000000  ...          0.000000    12.000000
75%     2054.140036           1.000000  ...          0.142857    12.000000
max    19043.138560           1.000000  ...          1.000000    12.000000

[8 rows x 17 columns]"""

print(df.isnull().sum())
"""Output:
    CUST_ID                             0
    BALANCE                             0
    BALANCE_FREQUENCY                   0
    PURCHASES                           0
    ONEOFF_PURCHASES                    0
    INSTALLMENTS_PURCHASES              0
    CASH_ADVANCE                        0
    PURCHASES_FREQUENCY                 0
    ONEOFF_PURCHASES_FREQUENCY          0
    PURCHASES_INSTALLMENTS_FREQUENCY    0
    CASH_ADVANCE_FREQUENCY              0
    CASH_ADVANCE_TRX                    0
    PURCHASES_TRX                       0
    CREDIT_LIMIT                        0
    PAYMENTS                            0
    MINIMUM_PAYMENTS                    0
    PRC_FULL_PAYMENT                    0
    TENURE                              0
    dtype: int64"""
    
df.fillna(method='ffill', inplace=True)

# Scaling
sc = StandardScaler()
df_sc = sc.fit_transform(df.iloc[:,1:])

# Data Visualization
sns.histplot(data=df, x='BALANCE')
plt.title('Distribution of Balance')
plt.show()

sns.histplot(data=df, x='PURCHASES')
plt.title('Distribution of Purchases')
plt.show()

sns.histplot(data=df, x='CASH_ADVANCE')
plt.title('Distribution of Cash Advances')
plt.show()

# Based on Income Levels
b = [0, 50000, 100000, 150000, 200000, np.inf]
lab = ['<50K', '50K-100K', '100K-150K', '150K-200K', '>200K']
df['INCOME_LEVEL'] = pd.cut(df['PURCHASES']/df['TENURE']*12, bins=b, labels=lab)

plt.figure(figsize=(12,6))
sns.histplot(x='BALANCE', hue='INCOME_LEVEL', data=df, kde=True)
plt.title('Balance Distribution for different Income Levels')
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(x='PURCHASES', hue='INCOME_LEVEL', data=df, kde=True)
plt.title('Purchases Distribution for different Income Levels')
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='INCOME_LEVEL', y='BALANCE', data=df)
plt.title('Balance Distribution for different Income Levels')
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='INCOME_LEVEL', y='PURCHASES', data=df)
plt.title('Purchases Distribution for different Income Levels')
plt.show()

plt.hist(df['CREDIT_LIMIT'], bins=20)
plt.title('Credit Limit Distribution')
plt.xlabel('Credit Limit')
plt.ylabel('Frequency')
plt.show()

# Clustering

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_sc)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

km = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
py = kmeans.fit_predict(df_sc)

df['Cluster'] = py
# Analyzing 
df_cluster_0 = df[df['Cluster']==0]
df_cluster_1 = df[df['Cluster']==1]
df_cluster_2 = df[df['Cluster']==2]

print('Cluster Sizes:')
print('Cluster 0:', len(df_cluster_0))
print('Cluster 1:', len(df_cluster_1))
print('Cluster 2:', len(df_cluster_2))
"""Output:
    Cluster Sizes:
    Cluster 0: 6123
    Cluster 1: 1589
    Cluster 2: 1238"""

print('\nCluster Means:')
print(df.groupby('Cluster').mean())
"""Output:
             BALANCE  BALANCE_FREQUENCY  ...  PRC_FULL_PAYMENT     TENURE
Cluster                                  ...                             
0         803.114798           0.835185  ...          0.155648  11.479993
1        3980.651090           0.958220  ...          0.033923  11.344871
2        2228.855587           0.981522  ...          0.297908  11.923263"""

print('\nCustom Analysis:')
print('Cluster 0:', df_cluster_0['CREDIT_LIMIT'].sum())
print('Cluster 1:', df_cluster_1['CREDIT_LIMIT'].sum())
print('Cluster 2:', df_cluster_2['CREDIT_LIMIT'].sum())
"""Output:
    Custom Analysis:
    Cluster 0: 20042773.585858002
    Cluster 1: 10606604.545455
    Cluster 2: 9575450.0"""

# Visulization
# 1. Hierarchical cluster
Z = linkage(df_sc, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.show()

# 2. DBSCAN cluster
db = DBSCAN(eps=0.5, min_samples=5)
clust = db.fit_predict(df_sc)
plt.scatter(df_sc[:,0], df_sc[:,1], c=clust, cmap='rainbow')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# 3. Spectral cluster
sp_c = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', assign_labels='kmeans')
sp_c = sp_c.fit_predict(df_sc)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_sc[:, 0], df_sc[:, 1], df_sc[:, 2], c=sp_c, cmap='rainbow')
plt.show()

from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# silhouette
km = KMeans(n_clusters=3)
km.fit(df_sc)
labels = km.labels_
sil = silhouette_samples(df_sc, km.labels_)
avg = silhouette_score(df_sc, km.labels_)
fig, ax = plt.subplots(figsize=(8,6))
y_lower, y_upper = 0, 0
for i, cl in enumerate(np.unique(km.labels_)):
    cl_vals = sil[km.labels_ == cl]
    cl_vals.sort()
    y_upper += len(cl_vals)
    col = cm.nipy_spectral(float(i) / len(np.unique(km.labels_)))
    ax.barh(range(y_lower, y_upper), cl_vals, height=1.0,
            edgecolor='none', color=col)
    y_lower += len(cl_vals)
ax.axvline(avg, color="red", linestyle="--")
ax.set_yticks([])
ax.set_xlim([-0.1, 1.0])
ax.set_xlabel("Silhouette coefficient values")
ax.set_ylabel("Cluster label")

plt.show()

############################################################################################################
#      THE DOCUMENTATION AND ALL GRAPHS ARE SEPERATELY UPLOADED AS A PDF FILE (P3_CLUSTER_ANALYSIS)        #
############################################################################################################