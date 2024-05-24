#include <iostream>
#include <vector>
#include "kmeans.hpp"
#include <random>


std::vector<std::vector<double>> generate_data(
        int true_num_clusters,
        int points_per_cluster,
        int num_dimensions,
        int random_state
    ) {
    std::vector<std::vector<double>> data;
    std::mt19937 gen(random_state);
    std::uniform_real_distribution<double> cluster_center(0.0, 20.0);
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < true_num_clusters; ++i) {
        std::vector<double> center(num_dimensions);
        for (int j = 0; j < num_dimensions; ++j) {
            center[j] = cluster_center(gen);
        }

        for (int j = 0; j < points_per_cluster; ++j) {
            std::vector<double> point(num_dimensions);
            for (int k = 0; k < num_dimensions; ++k) {
                point[k] = center[k] + distribution(gen);
            }
            data.push_back(point);
        }
    }

    return data;
}

int main() {

    // Generate some random data
    int true_num_clusters = 3;
    int points_per_cluster = 100;
    int num_dimensions = 30;
    int random_state = 42;

    std::vector<std::vector<double>> data = generate_data(true_num_clusters, points_per_cluster, num_dimensions, random_state);

    // Number of clusters and maximum iterations
    int k = 3;
    int max_iterations = 100;

    // Create a KMeans object
    KMeans kmeans(k, max_iterations);

    // Fit the KMeans algorithm to the data
    kmeans.fit(data);

    // Get the cluster assignments for the data
    std::vector<int> labels = kmeans.predict(data);

    // Get the centroids of the clusters
    std::vector<std::vector<double>> centroids = kmeans.get_centroids();

    // Output the results
    std::cout << "Cluster assignments:\n";
    for (size_t i = 0; i < labels.size(); ++i) {
        std::cout << "Point " << i << " -> Cluster " << labels[i] << "\n";
    }

    std::cout << "\nCentroids:\n";
    for (size_t i = 0; i < centroids.size(); ++i) {
        std::cout << "Cluster " << i << " centroid: (";
        for (size_t j = 0; j < centroids[i].size(); ++j) {
            std::cout << centroids[i][j];
            if (j < centroids[i].size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")\n";
    }

    return 0;
}
