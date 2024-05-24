#include "kmeans.hpp"
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>

/**
 * @brief Construct a new KMeans::KMeans object
 * 
 * @param k Number of clusters
 * @param max_iterations Maximum number of iterations for the algorithm
 * @param random_state  Seed for the random number generator 
 */
KMeans::KMeans(int k, int max_iterations, int random_state = 42) : 
    k(k), max_iterations(max_iterations), random_state(random_state), rng(random_state) {
}


/**
 * @brief Fit the KMeans model to the data
 * 
 * @param data Observed data points
 */
void KMeans::fit(const std::vector<std::vector<double>>& data) {
    initialize_centroids(data);

    //iteratively assign clusters and update centroids
    for (int iter =0; iter < max_iterations; ++iter){
        //E-step: assign clusters
        std::vector<int> labels = assign_clusters(data);
        //M-step: update centroids
        update_centroids(data, labels);
    }
}

/**
 * @brief Predict the cluster assignments for new data points
 * 
 * @param data Observed data points
 * @return std::vector <int> Cluster assignments for the data points
 */
std::vector <int> KMeans::predict(const std::vector<std::vector<double>>& data){
    return assign_clusters(data);
}


/**
 * @brief Get the centroids of the clusters
 * 
 * @return std::vector<std::vector<double>> Centroids of the clusters
 */
std::vector<std::vector<double>> KMeans::get_centroids() const {
    return centroids;
}

/**
 * @brief Assign data points to clusters according to the distance to the
 * current centroids
 * 
 * @param data Observed data points
 * @return std::vector<int> Vector of cluster assignments
 */
std::vector<int> assign_clusters(const std::vector<std::vector<double>>& data){
    std::vector<int> labels(data.size());
    //loop through all clusters and assign to cluster with minimal distance
    for (size_t i = 0; i < data.size(), ++i) {
        double min_distance = std::numeric_limits<double>::max();
        int closest_centroid = 0;
        for (int j=0; j <  k; ++j) {
            double distance = calculate_distance(data[j], centroids[j]);
            //update minimal distance and best cluster assignment
            if distance < min_distance {
                min_distance = distance;
                closest_centroid = j;
            } 
        }
        labels[i] = closest_centroid;
    }
}


/**
 * @brief Update the centroids of the clusters for the current cluster assignments
 * 
 * @param data Observed data points
 * @param labels Current cluster assignments
 */
void update_centroids(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    //initialize counts and centroids with zeros:
    std::vector<int> counts(k, 0);
    centroids = std::vector<std::vector<double>>(k, std::vector<double>(data[0].size(), 0.0)); //k x num_features matrix

    //iterate over rows:
    for (size_t i = 0; i < data.size(); ++i) {
        int cluster = labels[i];
        counts[cluster] += 1;
        //iterate over columns:
        for (size_t j = 0; j < data[i].size(); ++j){
            centroids[cluster][j] += data[i][j]
        }
    }

    //calculate center of mass (i.e. mean) for every cluster:
    for (int i = 0; i < k; ++i) {
        for (size_t j = 0; j < centroids[i].size(); ++j){
            centroids[i][j] /= counts[i]
        }
    }
}

/**
 * @brief Initialize the centroids of the clusters
 * 
 * @param data Observed data points
 */
void KMeans::initialize_centroids(const std::vector<std::vector<double>>& data) {
    centroids.clear();

    //ensure there are enough data points for k clusters:
    if (data.size() < k) {
        std::cerr << "Error: Number of data points is less than the number of clusters\n";
        return;
    }

    //Generate list of indices and suffle them:
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int i = 0; i < k; ++i) {
        //randomly choose initial centroids from original data points
        centroids.push_back(data[indices[i]])
    }
    
}


/**
 * @brief Calculate the Euclidean distance between two points
 * 
 * @param point1 Vector representing the first point
 * @param point2 Vector representing the second point
 * @return double Euclidean distance between the two points
 */
double KMeans::calculate_distance(const std::vector<double>& point1, const std::vector<double>& point2) {
    double distance = 0.0;
    //size_t because unsigned int
    for (size_t i = 0; i< point1.size(), ++i) {
        //calculate the squared difference between each dimension
        distance += std::pow(point1[i] - point2[i], 2);
    }
    return std::sqrt(distance);
}