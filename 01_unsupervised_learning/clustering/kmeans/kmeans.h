#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

class KMeans {
public:
    KMeans(int k, int max_iterations);

    // Method to fit the model to the data
    void fit(const std::vector<std::vector<double>>& data);

    // Method to predict the cluster for new data points
    std::vector<int> predict(const std::vector<std::vector<double>>& data);

    // Get the centroids of the clusters
    std::vector<std::vector<double>> get_centroids() const;

private:
    int k; // Number of clusters
    int max_iterations; // Maximum number of iterations
    std::vector<std::vector<double>> centroids; // Centroids of the clusters

    // Method to initialize centroids
    void initialize_centroids(const std::vector<std::vector<double>>& data);

    // Method to assign clusters
    std::vector<int> assign_clusters(const std::vector<std::vector<double>>& data);

    // Method to update centroids
    void update_centroids(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);

    // Method to calculate the distance between two points
    double calculate_distance(const std::vector<double>& point1, const std::vector<double>& point2);
};

#endif // KMEANS_H
