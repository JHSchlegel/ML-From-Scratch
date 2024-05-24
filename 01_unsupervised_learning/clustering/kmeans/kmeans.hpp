#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <random>


/**
 * @brief KMeans class for clustering data points
 * 
 */
class KMeans {
public:
    /**
     * @brief Construct a new KMeans object
     * 
     * @param k Number of clusters
     * @param max_iterations Maximum number of iterations for the algorithm
     * @param random_state  Seed for the random number generator 
     */
    KMeans(int k, int max_iterations, int random_state = 42);

    /**
     * @brief Fit the KMeans model to the data
     * 
     * @param data Observed data points
     */
    void fit(const std::vector<std::vector<double>>& data);

    /**
     * @brief Predict the cluster assignments for new data points
     * 
     * @param data Observed data points
     * @return std::vector <int> Cluster assignments for the data points
     */
    std::vector<int> predict(const std::vector<std::vector<double>>& data);

    /**
     * @brief Get the centroids of the clusters
     * 
     * @return std::vector<std::vector<double>> Centroids of the clusters
     */
    std::vector<std::vector<double>> get_centroids() const;

private:
    int k; // Number of clusters
    int max_iterations; // Maximum number of iterations
    int random_state; // Seed for the random number generator
    std::mt19937 rng; // Random number generator
    std::vector<std::vector<double>> centroids; // Centroids of the clusters

    /**
     * @brief Initialize the centroids of the clusters
     * 
     * @param data Observed data points
     */
    void initialize_centroids(const std::vector<std::vector<double>>& data);

    /**
     * @brief Assign data points to clusters
     * 
     * @param data Observed data points
     * @return std::vector<int> Cluster assignments for the data points
     */
    std::vector<int> assign_clusters(const std::vector<std::vector<double>>& data);

    /**
     * @brief Update the centroids of the clusters
     * 
     * @param data Observed data points
     * @param labels Cluster assignments for the data points
     */
    void update_centroids(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);

    /**
     * @brief Calculate the Euclidean distance between two points
     * 
     * @param point1 First point
     * @param point2 Second point
     * @return double Euclidean distance between the two points
     */
    double calculate_distance(const std::vector<double>& point1, const std::vector<double>& point2);
};

#endif // KMEANS_H
