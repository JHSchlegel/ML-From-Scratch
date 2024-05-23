#include "kmeans.h"
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <iostream>

KMeans::KMeans(int k, int max_iterations) : k(k), max_iterations(max_iterations) {
    std::srand(std::time(0)); // Seed the random number generator
}

void KMeans::fit(const std::vector<std::vector<double>>& data) {
    initialize_centroids(data);

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::vector<int> labels = assign_clusters(data);
        update_centroids(data, labels);
    }
}

std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) {
    return assign_clusters(data);
}

std::vector<std::vector<double>> KMeans::get_centroids() const {
    return centroids;
}

void KMeans::initialize_centroids(const std::vector<std::vector<double>>& data) {
    centroids.clear();
    for (int i = 0; i < k; ++i) {
        centroids.push_back(data[std::rand() % data.size()]);
    }
}

std::vector<int> KMeans::assign_clusters(const std::vector<std::vector<double>>& data) {
    std::vector<int> labels(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        double min_distance = std::numeric_limits<double>::max();
        int best_cluster = 0;

        for (int j = 0; j < k; ++j) {
            double distance = calculate_distance(data[i], centroids[j]);
            if (distance < min_distance) {
                min_distance = distance;
                best_cluster = j;
            }
        }
        labels[i] = best_cluster;
    }
    return labels;
}

void KMeans::update_centroids(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    std::vector<int> counts(k, 0);
    centroids = std::vector<std::vector<double>>(k, std::vector<double>(data[0].size(), 0.0));

    for (size_t i = 0; i < data.size(); ++i) {
        int cluster = labels[i];
        counts[cluster]++;
        for (size_t j = 0; j < data[i].size(); ++j) {
            centroids[cluster][j] += data[i][j];
        }
    }

    for (int i = 0; i < k; ++i) {
        for (size_t j = 0; j < centroids[i].size(); ++j) {
            centroids[i][j] /= counts[i];
        }
    }
}

double KMeans::calculate_distance(const std::vector<double>& point1, const std::vector<double>& point2) {
    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        sum += std::pow(point1[i] - point2[i], 2);
    }
    return std::sqrt(sum);
}
