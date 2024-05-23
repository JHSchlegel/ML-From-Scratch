#include <iostream>
#include <vector>
#include "kmeans.h"

int main() {
    // Example data: 2D points
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {1.5, 1.8},
        {5.0, 8.0},
        {8.0, 8.0},
        {1.0, 0.6},
        {9.0, 11.0},
        {8.0, 2.0},
        {10.0, 2.0},
        {9.0, 3.0}
    };

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
