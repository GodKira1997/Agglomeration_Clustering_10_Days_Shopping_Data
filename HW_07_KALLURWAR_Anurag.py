"""
file: HW_07_KALLURWAR_Anurag.py
description: This program performs agglomerative hierarchichal clustering on
the supermarket dataset.
language: python3
author: Anurag Kallurwar, ak6491@rit.edu
"""


import warnings
import sys
import os
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt


# CLASSES
class Prototype():
    """
    This class represents a cluster prototypes of the data
    """
    # Class Variables
    __slots__ = "label", "members", "center"

    # Class Methods
    def __init__(self, label, members: list):
        """
        Constructor
        :param cluster_center: Best attribute used for splitting
        :param members: Member datapoints of the cluster
        """
        self.label = label
        self.members = members
        self.center = None

    def get_size(self):
        """
        Gets the size of the prototype
        :return: int size
        """
        return len(self.members)

    def calculate_center_of_mass(self, reduced_features: list):
        """
        Calculates the cetroid for the prototype
        :return: None
        """
        members = []
        for member in self.members:
            member_array = [member[feature] for feature in reduced_features]
            members.append(np.array(member_array))
        self.center = np.mean(members, axis=0)

    def __str__(self):
        """
        String Format output for the class
        :return: str
        """
        return str(self.label) + " | Size: " + str(self.get_size())

    def __repr__(self):
        """
        Detailed (Representative) String Format output for the class
        :return: str
        """
        return str(self.label) + " | Centroid: " + str(np.round(self.center,
                                                                decimals=3))


# FUNCTIONS
def clean_data(df: pd.DataFrame):
    """
    Cleaning the dataframe
    :param df: input dataframe
    :return: cleaned dataframe
    """
    return df.dropna()


def read_file(file_name: str):
    """
    Read the CSV file and return dataframe
    :param file_name: filename
    :return: dataframe
    """
    print("Reading file: " + file_name)
    dataframe = pd.read_csv(file_name, low_memory=False)
    dataframe = clean_data(dataframe)
    return dataframe


def write_file(file_name: str, dataframe: pd.DataFrame):
    """
    Write the CSV file and return dataframe
    :param file_name: filename
    :param dataframe: Output datafram
    :return: None
    """
    print("Writing to file: " + file_name)
    dataframe.to_csv(file_name)


def reduce_features(supermarket_df: pd.DataFrame):
    """
    Searches for the irrevelant features by utilizing the cross-correlation
    coefficients between the features
    :param supermarket_df: Input dataframe
    :return: cross-correlation matrix, max coefficient for every feature,
    irrelevant features
    """
    # Drop column ID
    features_to_be_dropped = ['ID']
    # Creating the correlation matrix for all features except 'ID'
    reduced_features = supermarket_df.columns.difference(
        features_to_be_dropped).tolist()
    correlation_matrix = supermarket_df[reduced_features].corr().round(2)
    correlation_matrix_absolute = correlation_matrix.abs()
    # Calculate Max correlation coefficient for every feature
    max_correlation_features = []
    for feature in correlation_matrix_absolute.columns:
        other_features = correlation_matrix_absolute[feature].drop([
            feature]).index
        max_correlation_feature = correlation_matrix_absolute[feature][
            other_features].idxmax()
        max_correlation_features.append([feature, max_correlation_feature,
                                  correlation_matrix_absolute.loc[feature,
                                                         max_correlation_feature]])
    # Get 4 with minimum "maximum correlations"
    max_correlation_features = sorted(max_correlation_features, key=lambda x:
    x[2])
    features_to_be_dropped += [x[0] for x in max_correlation_features[:4]]
    # features_to_be_dropped = ['  Salt', ' Fruit', ' Beans', '  Eggs']
    return correlation_matrix, max_correlation_features, features_to_be_dropped


def calculate_euclidean_distance(cluster1: Prototype, cluster2: Prototype):
    """
    Calculates the euclidean distance between two prototypes
    :param cluster1: Prototype object
    :param cluster2: Prototype object
    :return: square of the euclidean distance
    """
    # Calculating the square of distance between centroids of the clusters
    return ((cluster1.center - cluster2.center) ** 2).sum()


def initialize_clusters(supermarket_df: pd.DataFrame, reduced_features: list):
    """
    Initiliaze all clusters with almost one datapoint
    :param supermarket_df: Input dataframe
    :param reduced_features: list of features selected for calculation
    :return: dictionary of clusters
    """
    clusters = dict()
    for index, row in supermarket_df.iterrows():
        # 1 cluster for 1 datapoint
        cluster = Prototype(index, [row])
        # Calculate the centroid for the cluster
        cluster.calculate_center_of_mass(reduced_features)
        # Putting cluster to dictionary
        clusters[cluster.label] = cluster
    return clusters


def initialize_distances(clusters: dict()):
    """
    Calculate distance matrix for intial clusters
    :param supermarket_df: Input dataframe
    :param reduced_features: list of features selected for calculation
    :return: dictionary of distance matrix
    """
    distance_matrix = dict()
    # Getting all cluster labels
    cluster_labels = list(clusters.keys())
    # Looping over all labels
    for index1 in range(len(cluster_labels)):
        cluster_label1 = cluster_labels[index1]
        # Looping over all labels except current label
        for index2 in range(index1 + 1, len(cluster_labels)):
            cluster_label2 = cluster_labels[index2]
            # Euclidean distance between clusters
            distance = calculate_euclidean_distance(clusters[cluster_label1],
                                                    clusters[cluster_label2])
            # Putting distance to dictionary
            distance_matrix[(cluster_label1, cluster_label2)] = distance
    return distance_matrix


def perform_agglomerative_clustering(supermarket_df: pd.DataFrame,
                                     reduced_features: list):
    """
    Initialize clusters and distance matrix and performs agglomerative
    hierarchical clustering on the clusters
    :param supermarket_df: Input dataframe
    :param reduced_features: list of features selected for calculation
    :return:
    """
    # Initial clusters and distance matrix
    print("\nInitializing...")
    clusters = initialize_clusters(supermarket_df, reduced_features)
    distance_matrix = initialize_distances(clusters)
    # print(distance_matrix)
    # print(len(distance_matrix.keys()))

    # List to store the last 20 cluster merged
    cluster_mergers = []
    print("\nStarting Hierarchical Clustering...")
    # Loop until 1 cluster is left
    while len(clusters) > 1:
        print("\tCluster count: ", len(clusters))

        # Calculating the closest clusters
        closest_clusters = min(distance_matrix, key=distance_matrix.get)
        cluster_label1 = min(closest_clusters)
        cluster_label2 = max(closest_clusters)

        # Combining the closest clusters and assigning the minimum of
        # closest cluster labels to the new combined cluster
        combined_members = clusters[cluster_label1].members + clusters[
            cluster_label2].members
        combined_cluster = Prototype(cluster_label1, combined_members)
        combined_cluster.calculate_center_of_mass(reduced_features)

        # Storing the last 20 merged clusters
        if len(clusters) <= 21:
            cluster_mergers.append([distance_matrix[closest_clusters],
                                     (cluster_label1, clusters[
                                         cluster_label1]), (cluster_label2,
                                                            clusters[cluster_label2])])

        # Updating the dictionary of clusters with new cluster
        clusters[combined_cluster.label] = combined_cluster
        # Removing for data for the clusters combined from both dictionaries
        del clusters[cluster_label2]
        for key in list(distance_matrix.keys()):
            if cluster_label1 in key or cluster_label2 in key:
                del distance_matrix[key]

        # Calculating distances for the new cluster
        combined_label = combined_cluster.label
        for cluster_label in list(clusters.keys()):
            if cluster_label != combined_label:
                distance = calculate_euclidean_distance(clusters[
                                                            combined_label],
                                                        clusters[cluster_label])
                distance_matrix[(combined_label, cluster_label)] = distance
        # print(len(distance_matrix))
    return cluster_mergers


def create_dendogram(supermarket_df: pd.DataFrame, reduced_features: list):
    """
    Creates a dendogram for the input dataset
    :param supermarket_df: input dataframe
    :param reduced_features: list of features selected for calculation
    :return: None
    """
    print("")
    # Distance matrix for the data based on selected features
    distance_matrix = hierarchy.distance.pdist(supermarket_df[reduced_features],
                                               metric='euclidean')
    # Linkage Matrix based on euclidean distance matrix of data points
    linkage_matrix = hierarchy.linkage(distance_matrix, method='average')
    # Dendrogram based on linkage matrix (actaully centroid)
    dendrogram = hierarchy.dendrogram(linkage_matrix,
                                      labels=supermarket_df.index,
                                      orientation='top',
                                      truncate_mode='lastp', p=20,
                                      show_contracted=True)
    # Display the dendrogram
    print("\nPlotting Dendogram plot...")
    print("Close the window to continue!")
    plt.suptitle("Dendrogram")
    plt.title("Agglomerative Clustering with Centroid Linkage for "
              "'supermarket_df'", fontsize=9)
    plt.tick_params(axis='both', labelsize=6)
    plt.xlabel('Data Points', fontsize=7)
    plt.ylabel('Distances', fontsize=7)
    plt.show()


def process(supermarket_df: pd.DataFrame):
    """
    This function compute revelant features and performs agglomerative
    clustering to find the inherent structure of the data
    :param supermarket_df: Input dataframe
    :return: None
    """
    # Part 1.
    # Feature Selection
    print("\n\n========================= Feature Selection "
          "=========================")
    # Correlation coefficiants
    correlation_matrix, max_correlation_features, features_to_be_dropped = \
        reduce_features(supermarket_df)
    write_file("correlation_matrix.csv" ,correlation_matrix)

    # Printing outputs for the questions
    print("\n-- Report --")
    print("Most strongly cross-correlated:", end=" ")
    print(max_correlation_features[-1])
    print("Cross-correlation coefficient of Chips with Cereal:", end=" ")
    print(correlation_matrix[' Chips']['Cereal'])
    for correlation in max_correlation_features:
        if 'Fish' in correlation[0]:
            print("Fish most cross-correlated with:", end=" ")
            print(correlation[1], correlation_matrix[correlation[0]][
                correlation[1]])
        if 'Chips' in correlation[0]:
            print("Chips most cross-correlated with:", end=" ")
            print(correlation[1], correlation_matrix[correlation[0]][
                correlation[1]])
            print("Cross-correlation coefficient of Fish and ChildBby:",
                  end=" ")
            print(correlation_matrix['  Fish']['ChildBby'])
        if 'Vegges' in correlation[0]:
            print("Vegges most cross-correlated with:", end=" ")
            print(correlation[1], correlation_matrix[correlation[0]][
                correlation[1]])
    print("Cross-correlation coefficient of Milk with Cereal:", end=" ")
    print(correlation_matrix['  Milk']['Cereal'])
    print("Two attributes are not strongly cross-correlated with anything:",
          end=" ")
    print(str(max_correlation_features[0][0]) + ", " + str(
        max_correlation_features[1][0]))
    print("Which attributes would you believe were irrelevant:", end=" ")
    print(features_to_be_dropped)
    # Using irrelevant features for feature selection
    print("Features selected for clustering:")
    reduced_features = supermarket_df.columns.difference(
        features_to_be_dropped).tolist()
    print(reduced_features)

    # Part 2.
    # Agglomerative Clustering
    print("\n\n========================= Agglomerative Clustering "
          "=========================")
    cluster_mergers = perform_agglomerative_clustering(supermarket_df,
                                                       reduced_features)
    # Analysis based on last 20 clusters merged
    print("\n\n========================= Clusters Analysis "
          "=========================")
    print("\n-- Last 20 mergers --")
    for index in range(1, len(cluster_mergers) + 1):
        print(str(index) + ": " + str(cluster_mergers[-index][1][0]) + ", "
              + str(cluster_mergers[-index][2][0]) + " | Sizes: "
              + str(cluster_mergers[-index][1][1].get_size()) + ", "
              + str(cluster_mergers[-index][2][1].get_size()) + " | Distance: "
              + str(round(cluster_mergers[-index][0], 3)))

    print("\n-- Last 20 smallest clusters merged --")
    for index in range(1, len(cluster_mergers) + 1):
        print(str(index) + ": " + str(cluster_mergers[-index][1][0])
              + " | Size: " + str(cluster_mergers[-index][1][1].get_size()))

    # Computed optimal clusters based on last 20 clusters merged
    optimal_clusters = [cluster_mergers[-1][2][1]] + [cluster_mergers[-2][2][
                                                          1]] + [
        cluster_mergers[-3][1][1]] + [cluster_mergers[-3][2][1]]
    optimal_clusters = sorted(optimal_clusters, key=lambda x: x.get_size())
    print("\n-- Optimal clusters --")
    print("Cluster Sizes:")
    for cluster in optimal_clusters:
        print("Cluster: " + str(cluster))
    print("Cluster Centroids:")
    for cluster in optimal_clusters:
        print("Cluster: " + repr(cluster))


    # Part 3.
    # Dendogram
    print("\n\n========================= Dendogram "
          "=========================")
    create_dendogram(supermarket_df, reduced_features)


def main():
    """
    The main function
    :return: None
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if len(sys.argv) < 2:
        print("Missing Argument!")
        print("Usage: HW_07_KALLURWAR_Anurag.py <filename.csv>")
        return
    file_name = sys.argv[1].strip()
    if not os.path.isfile(os.getcwd() + "\\" + file_name):
        print("Please put " + file_name + " in the execution folder!")
        return
    supermarket_df = read_file(file_name)
    print(supermarket_df)
    process(supermarket_df)


if __name__ == '__main__':
    main()  # Calling Main Function
