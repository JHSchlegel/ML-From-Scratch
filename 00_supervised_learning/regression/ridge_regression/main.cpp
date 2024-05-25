#include <iostream>
#include <Eigen/Dense>
#include <random>
#include "ridge_reg.hpp" 

/**
 * @brief Generate synthetic data for linear regression
 * 
 * @param X Feature matrix to be generated
 * @param y Target vector to be generated
 * @param n_samples Number of samples to generate
 * @param true_params True coefficients of the model
 * @param noise_level Noise level for the target variable
 */
void generateData(Eigen::MatrixXd& X, Eigen::VectorXd& y, int n_samples, const Eigen::VectorXd& true_params, double noise_level) {
    int random_state = 42;
    std::mt19937 gen(random_state);
    std::normal_distribution<> d(0, noise_level);
    std::normal_distribution<> feature_dist(5, 10);

    int n_features = true_params.size();
    X.resize(n_samples, n_features);
    y.resize(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            if (j == 0) X(i, j) = 1; // bias term
            else X(i, j) = feature_dist(gen);
        }
        y(i) = true_params.dot(X.row(i)) + d(gen); // generate y with noise
    }
}

/**
 * @brief Calculate the Mean Squared Error between the true and predicted values
 * 
 * @param y_true True target values
 * @param y_pred Predicted target values
 * @return double Mean Squared Error between the true and predicted values
 */
double calculateMSE(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    return (y_true - y_pred).array().square().mean();
}

int main() {
    int n_train = 1000;
    int n_test = 200;
    double noise_level = 5.0;

    // Number of features
    int n_features = 100;

    // True parameters for the model
    Eigen::VectorXd true_params = Eigen::VectorXd::Random(n_features);

    // Generate training data
    Eigen::MatrixXd X_train;
    Eigen::VectorXd y_train;
    generateData(X_train, y_train, n_train, true_params, noise_level);

    // Generate testing data
    Eigen::MatrixXd X_test;
    Eigen::VectorXd y_test;
    generateData(X_test, y_test, n_test, true_params, noise_level);

    // Train the Ridge regression model
    RidgeReg model;
    model.fit(X_train, y_train);

    // Predict on the test set
    Eigen::VectorXd y_pred = model.predict(X_test);

    // Calculate the Mean Squared Error on the test set
    double mse = calculateMSE(y_test, y_pred);

    std::cout << "True coefficients:\n" << true_params << "\n";
    std::cout << "Estimated coefficients (beta):\n" << model.get_coefficients() << "\n";
    std::cout << "Number of zero coefficients: " << (model.get_coefficients().array().abs() < 1e-20).count() << "\n";
    std::cout << "Mean Squared Error on the test set: " << mse << "\n";

    return 0;
}
