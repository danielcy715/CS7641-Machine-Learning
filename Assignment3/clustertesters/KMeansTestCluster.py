

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import metrics, decomposition
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


class KMeansTestCluster():
    def __init__(self, X, y, clusters, plot=True, targetcluster=3, stats=False):
        self.X = X
        self.y = y
        self.clusters = clusters
        self.gen_plot = plot
        self.targetcluster = targetcluster
        self.stats = stats

    def run(self):
        meandist=[]
        homogeneity_scores=[]
        completeness_scores=[]
        v_measure = []
        rand_scores=[]
        silhouettes=[]

        for k in self.clusters:
            model = KMeans(n_clusters=k, max_iter=500, init='k-means++')
            labels = model.fit_predict(self.X)
            #print model.cluster_centers_
            #print cdist(self.X, model.cluster_centers_, 'euclidean')
            #print cdist(self.X, model.cluster_centers_, 'euclidean').shape
            if k == self.targetcluster and self.stats:
                #print labels
                nd_data = np.concatenate((self.X, np.expand_dims(labels, axis=1),np.expand_dims(self.y, axis=1)), axis=1)
                pd_data = pd.DataFrame(nd_data)
                pd_data.to_csv("cluster_kmeans.csv", index=False, index_label=False, header=False)
                #print model.cluster_centers_
                print cdist(self.X, model.cluster_centers_, 'euclidean').shape

            #print np.min(np.square(cdist(self.X, model.cluster_centers_, 'euclidean')), axis = 1)
            min = np.min(np.square(cdist(self.X, model.cluster_centers_, 'euclidean')), axis = 1)
            print "###"
            print -model.score(self.X)/self.X.shape[0]
            #print min
            value = np.mean(min)
            meandist.append(value)

            homogeneity_scores.append(metrics.homogeneity_score(self.y, labels))
            completeness_scores.append(metrics.completeness_score(self.y, labels))
            rand_scores.append(metrics.adjusted_rand_score(self.y, labels))
        print meandist
        if self.gen_plot:
            #self.visualize()

            self.plot(meandist, homogeneity_scores, completeness_scores, rand_scores, silhouettes)

    def visualize(self):
        """
        Generate scatter plot of Kmeans with Centroids shown
        """
        fig = plt.figure(1)
        plt.clf()
        plt.cla()

        X_new = decomposition.pca.PCA(n_components=3).fit_transform(self.X)
        model = KMeans(n_clusters=self.targetcluster, max_iter=5000, init='k-means++')
        labels = model.fit_predict(X_new)
        totz = np.concatenate((X_new,  np.expand_dims(labels, axis=1), np.expand_dims(self.y, axis=1),), axis=1)

        # for each cluster
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for clust in range(0, self.targetcluster):
            totz_clust = totz[totz[:,-2] == clust]
            print "Cluster Size"
            print totz_clust.shape

            benign = totz_clust[totz_clust[:,-1] == 1]
            malignant = totz_clust[totz_clust[:,-1] == 0]

            ax.scatter(benign[:, 0], benign[:, 1], benign[:, 2], color=colors[clust], marker=".")
            ax.scatter(malignant[:, 0], malignant[:, 1], malignant[:, 2], color=colors[clust], marker="x")

        centroids = model.cluster_centers_
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
            marker='x', s=169, linewidths=3, color="black",
             zorder=10)

        # ax.title("Breast Cancer Clustering")
        ax.set_xlabel("1st Component")
        ax.set_ylabel("2nd Component")
        ax.set_zlabel("3rd Component")
        plt.show()

    def plot(self, meandist, homogeneity, completeness, rand, silhouettes):
            """
            Plot average distance from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, meandist)
            plt.xlabel('Number of clusters')
            plt.ylabel('Average within cluster SSE')
            plt.title('Breast Cancer-Kmeans-Average within cluster SSE')
            #plt.show()

            #plt.clf()

            """
            Plot homogeneity from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, homogeneity)
            plt.xlabel('Number of clusters')
            plt.ylabel('Homogeneity Score')
            plt.title('Breast Cancer-Kmeans-Homogeneity Score')
            #plt.show()

            #plt.clf()

            """
                       Plot completeness from observations from the cluster centroid
                       to use the Elbow Method to identify number of clusters to choose
                       """
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, completeness)
            plt.xlabel('Number of clusters')
            plt.ylabel('Completeness Score')
            plt.title('Breast Cancer-Kmeans-Completeness Score')
            plt.show()

            #plt.clf()

            """
            Plot Adjusted RAND Score from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose

            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, rand)
            plt.xlabel('Number of clusters')
            plt.ylabel('Adjusted RAND Score')
            plt.title('Breast Cancer-Kmeans-RAND Score')
            plt.show()
            """

