#Libaries
import os
os.environ["OMP_NUM_THREADS"] = "2"

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

#import dataset
#ensure that all null values are filled with 0
training = pd.read_csv('06 Training Clustering Data.csv')
complete_census = pd.read_csv('Prepared Clustering Data.csv')
codes = complete_census['CODE']

###########################
#    Defining Functions   #
###########################

def iterate_kmeans(k_cluster, maxK):
    #A list holds the SSE values for each k
    sse = []
    silhouette_coefficients = []
    for k in range(2, maxK):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(k_cluster)
        sse.append(kmeans.inertia_)
        sscore = silhouette_score(k_cluster, kmeans.labels_)
        silhouette_coefficients.append(sscore)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, maxK), silhouette_coefficients)
    plt.xticks(range(2, maxK))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()

#estabishing kmeans parameters before the loops
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 1000,
    "random_state": 42,
}
#############################
# Creating Primary Clusters #
#############################


cnFeatures = np.array(training.drop(['CODE'], 1).astype(float))
varnames = list(training.drop(['CODE'], 1).columns)
complete_cnFeatures = np.array(complete_census.drop(['CODE'], 1).astype(float))
complete_varnames = list(complete_census.drop(['CODE'], 1).columns)


#iterating through kmeans to determine optimal cluster
#Specify how many clusters to examine
iterate_kmeans(cnFeatures, 21)

####  K CLUSTERS #####
#number of clusters
#specify after checking the plots
num_clusters = 4

kmeans = KMeans(n_clusters=num_clusters, **kmeans_kwargs)
kmeans.fit(cnFeatures)
predict = kmeans.predict(cnFeatures)
complete_predict = kmeans.predict(complete_cnFeatures)


#create primary clusters dataframe
training = pd.DataFrame(cnFeatures)
training.columns = varnames
training['CODE'] = codes
training['cluster'] = predict
#create primary clusters dataframe for complete census
complete_census = pd.DataFrame(complete_cnFeatures)
complete_census.columns = complete_varnames
complete_census['CODE'] = codes
complete_census['cluster'] = complete_predict
#creating cluster centroids file
primary_centers = pd.DataFrame(kmeans.cluster_centers_)
primary_centers.columns = varnames
primary_centers.to_csv('primary_centers.csv')


###############################
# Creating Secondary Clusters #
###############################

#list of primary clusters
primary_clusters = training.cluster.unique()
primary_clusters.sort()

####  K CLUSTERS #####
####  Ensure that list is as long as initial clusters
###   Specify the number of subclusters per cluster
num_clusters = [4,3,2,3]

#empty list to store all cluster results
result = []
centroids = []

#Creating subclusters
for c in primary_clusters:
    print('cluster ', c)

    #filtering census equal to current cluster
    subcensus = training[training.cluster.eq(c)]
    subcensus.reset_index(drop=True, inplace=True)
    #filtering to the current cluster and setting columns
    complete_subcensus = complete_census[complete_census.cluster.eq(c)]
    complete_subcensus.reset_index(drop=True, inplace=True)
    complete_codes = complete_subcensus['CODE']
    complete_initial_cluster = complete_subcensus['cluster']
    print(len(complete_codes), 'areas', '-', num_clusters[c], ' secondary clusters')

    #creating array for current subcluster
    subFeatures = np.array(subcensus.drop(['CODE','cluster'], 1).astype(float))
    varnames = list(subcensus.drop(['CODE','cluster'], 1).columns)
    complete_subFeatures = np.array(complete_subcensus.drop(['CODE','cluster'], 1).astype(float))
    complete_varnames = list(complete_subcensus.drop(['CODE','cluster'], 1).columns)
    
    #iterating through kmeans to determine optimal cluster
    iterate_kmeans(subFeatures, 21)

    #kmeans clustering
    kmeans = KMeans(n_clusters=num_clusters[c], **kmeans_kwargs)
    kmeans.fit(subFeatures)

    #adding classification to census
    outclass = pd.DataFrame(complete_subFeatures)
    outclass.columns = complete_varnames
    outclass['CODE'] = complete_codes
    outclass['initial_cluster'] = complete_initial_cluster
    outclass['second_cluster'] = kmeans.predict(complete_subFeatures)
    #appending results to output list
    result.append(outclass)

    #creating cluster centers
    centers = pd.DataFrame(kmeans.cluster_centers_)
    centers.columns = varnames
    centers.index.name = 'cluster'
    centers = centers.reset_index().rename(columns={centers.index.name: 'cluster'})
    centers['cluster'] = [str(c) + '_' + str(i) for i in centers['cluster']]
    centroids.append(centers)

    
#writing final file
pd.concat(result).to_csv('final_clusters.csv', index=False)

#creating cluster centroids file
pd.concat(centroids).to_csv('secondary_centers.csv')


#END