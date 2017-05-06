"""
A K means experimenter
__author__      = "Jonathan Satria"
__date__ = "April 01, 2016"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, decomposition
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.ticker import MaxNLocator

class ExpectationMaximizationTestCluster():
    def __init__(self, X, y, clusters, plot=False, targetcluster=3, stats=False):
        self.X = X
        self.y = y
        self.clusters = clusters
        self.gen_plot = plot
        self.targetcluster = targetcluster
        self.stats = stats

    def run(self):
        ll=[]
        homogeneity_scores=[]
        completeness_scores=[]
        rand_scores=[]
        silhouettes=[]
        bic=[]
        aic=[]
        model = GMM(covariance_type = 'diag')

        for k in self.clusters:
            model.set_params(n_components=k)
            model.fit(self.X)
            labels = model.predict(self.X)
            #print labels
            if k == self.targetcluster and self.stats:
                nd_data = np.concatenate((self.X, np.expand_dims(labels, axis=1),np.expand_dims(self.y, axis=1)), axis=1)
                pd_data = pd.DataFrame(nd_data)
                pd_data.to_csv("cluster_em.csv", index=False, index_label=False, header=False)

                for i in range (0,self.targetcluster):
                    #print "Cluster {}".format(i)
                    cluster = pd_data.loc[pd_data.iloc[:,-2]==i].iloc[:,-2:]
                    print cluster.shape[0]
                    #print float(cluster.loc[cluster.iloc[:,-1]==0].shape[0])/cluster.shape[0]
                    #print float(cluster.loc[cluster.iloc[:,-1]==1].shape[0])/cluster.shape[0]

            #meandist.append(sum(np.min(cdist(self.X, model.cluster_centers_, 'euclidean'), axis=1))/ self.X.shape[0])
            ll.append(model.score(self.X))
            print model.score(self.X)
            homogeneity_scores.append(metrics.homogeneity_score(self.y, labels))
            completeness_scores.append(metrics.completeness_score(self.y, labels))
            rand_scores.append(metrics.adjusted_rand_score(self.y, labels))
            bic.append(model.bic(self.X))
            aic.append(model.aic(self.X))
            #silhouettes.append(metrics.silhouette_score(self.X, model.labels_ , metric='euclidean',sample_size=self.X.shape[0]))

        if self.gen_plot:
            #self.visualize()
            self.plot(ll, homogeneity_scores, completeness_scores, rand_scores, bic, aic)

    def visualize(self):
        """
        Generate scatter plot of Kmeans with Centroids shown
        """
        fig = plt.figure(1)
        plt.clf()
        plt.cla()

        X_new = decomposition.pca.PCA(n_components=2).fit_transform(self.X)
        model = GMM(n_components=self.targetcluster, covariance_type='full')
        labels = model.fit_predict(X_new)
        totz = np.concatenate((X_new,  np.expand_dims(labels, axis=1), np.expand_dims(self.y, axis=1),), axis=1)

        # for each cluster
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
        fig = plt.figure()

        for clust in range(0, self.targetcluster):
            totz_clust = totz[totz[:,-2] == clust]
            print "Cluster Size"
            print totz_clust.shape

            benign = totz_clust[totz_clust[:,-1] == 1]
            malignant = totz_clust[totz_clust[:,-1] == 0]

            plt.scatter(benign[:, 0], benign[:, 1],  color=colors[clust], marker=".")
            plt.scatter(malignant[:, 0], malignant[:, 1],  color=colors[clust], marker="x")

        # centroids = model.
        # plt.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
        #     marker='x', s=169, linewidths=3, color="black",
        #      zorder=10)

        plt.title("Breast Cancer Clustering")
        plt.xlabel("1st Component")
        plt.ylabel("2nd Component")
        plt.show()

    def plot(self, ll, homogeneity, completeness, rand, bic, aic):
            # """
            # Plot average distance from observations from the cluster centroid
            # to use the Elbow Method to identify number of clusters to choose
            # """
            # plt.plot(self.clusters, meandist)
            # plt.xlabel('Number of clusters')
            # plt.ylabel('Average distance')
            # plt.title('Average distance vs. K Clusters')
            # plt.show()
            #
            # plt.clf()

            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, ll)
            plt.xlabel('Number of clusters')
            plt.ylabel('Log Probability')
            plt.title('Letter Recognition-EM-Log Probability')
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
            plt.title('Letter Recognition-EM-Homogeneity Score')
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
            plt.title('Letter Recognition-EM-Completeness Score')
            #plt.show()

            #plt.clf()


            """
            Plot Adjusted RAND Score from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose

            plt.plot(self.clusters, rand)
            plt.xlabel('Number of clusters')
            plt.ylabel('Adjusted RAND Score')
            plt.title('RAND Score vs. K Clusters')
            plt.show()
            """
            """
            Plot BIC Score from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, bic)
            plt.xlabel('Number of clusters')
            plt.ylabel('BIC Score')
            plt.title('Letter Recognition-EM-BIC Score')
            #plt.show()

            """
            Plot AIC Score from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, aic)
            plt.xlabel('Number of clusters')
            plt.ylabel('AIC Score')
            plt.title('Letter Recognition-EM-AIC Score')
            plt.show()