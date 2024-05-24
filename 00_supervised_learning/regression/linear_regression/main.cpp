#include <iostream>
#include <Eigen/Dense>
#include <random>
#include "lin_reg.hpp" 

/**
 * @brief Generate synthetic data for linear regression
 * 
 * @param X Feature matrix to be generated
 * @param y Target vector to be generated
 * @param n_samples Number of samples to generate
 * @param true_params True coefficiets of the model
 * @param noise_level Noise level for the target variable
 */
void generateData(Eigen::MatrixXd& X, Eigen::VectorXd& y, int n_samples, const Eigen::VectorXd& true_params, double noise_level) {
    int random_state = 42;
    std::mt19937 gen(random_state);
    std::normal_distribution<> d(0, noise_level);
    std::normal_distribution<> feature_dist(0, 1);

    int n_features = true_params.size();
    X.resize(n_samples, n_features);
    y.resize(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        X(i, 0) = 1; // bias term
        double X_sample = feature_dist(gen);
        X(i, 1) = X_sample; // first feature
        if (n_features > 2) {
            X(i, 2) = X_sample * X_sample; // second feature (squared)
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
    int n_train = 500;
    int n_test = 100;
    double noise_level = 1.0;

    // True parameters for the model [intercept, linear coefficient, quadratic coefficient]
    Eigen::VectorXd true_params(3);
    true_params << 3.5, 2.0, 1.0;

    // Generate training data
    Eigen::MatrixXd X_train;
    Eigen::VectorXd y_train;
    generateData(X_train, y_train, n_train, true_params, noise_level);

    // Generate testing data
    Eigen::MatrixXd X_test;
    Eigen::VectorXd y_test;
    generateData(X_test, y_test, n_test, true_params, noise_level);

    // Train the linear regression model using QR decomposition
    LinReg model;
    model.fit(X_train, y_train);

    // Predict on the test set
    Eigen::VectorXd y_pred = model.predict(X_test);

    // Calculate the Mean Squared Error on the test set
    double mse = calculateMSE(y_test, y_pred);

    std::cout << "True coefficients:\n" << true_params << "\n";
    std::cout << "Estimated coefficients (beta):\n" << model.get_coefficients() << "\n";
    std::cout << "Mean Squared Error on the test set: " << mse << "\n";

    return 0;
}
