#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <cmath>

#include "lasso_reg.hpp"

/**
 * @brief Generate synthetic data for linear regression
 *
 * @param X Feature matrix to be generated
 * @param y Target vector to be generated
 * @param n_samples Number of samples to generate
 * @param true_params True coefficients of the model
 * @param noise_level Noise level for the target variable
 */
void generateData(Eigen::MatrixXd& X,
                  Eigen::VectorXd& y,
                  int n_samples,
                  const Eigen::VectorXd& true_params,
                  double noise_level)
{
    int random_state = 42;
    std::mt19937 gen(random_state);
    std::normal_distribution<> d(0, noise_level);
    std::normal_distribution<> feature_dist(5, 10);

    int n_features = true_params.size();
    X.resize(n_samples, n_features);
    y.resize(n_samples);

    for (int i = 0; i < n_samples; ++i)
    {
        for (int j = 0; j < n_features; ++j)
        {
            if (j == 0)
                X(i, j) = 1;  // bias term
            else
                X(i, j) = feature_dist(gen);  // generate features
        }
        y(i) = true_params.dot(X.row(i)) + d(gen);  // generate y with noise
    }
}

/**
 * @brief Calculate the Mean Squared Error between the true and predicted
 * values
 *
 * @param y_true True target values
 * @param y_pred Predicted target values
 * @return double Mean Squared Error between the true and predicted values
 */
double calculateMSE(const Eigen::VectorXd& y_true,
                    const Eigen::VectorXd& y_pred)
{
    return (y_true - y_pred).array().square().mean();
}

int main()
{
    int n_train = 1000;
    int n_test = 200;
    double noise_level = 5.0;

    // Number of features
    int n_features = 25;

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

    // Try different lambda values
    std::vector<double> lambdas = {0.1, 1.0, 10.0, 100.0};
    
    std::cout << "\nLasso Regression Results:\n";
    std::cout << "========================\n\n";
    
    for (double lambda : lambdas) {
        std::cout << "Lambda = " << lambda << ":\n";
        
        // Train the Lasso regression model
        LassoReg model(lambda, 1000, 1e-6);
        model.fit(X_train, y_train);
        
        // Predict on the test set
        Eigen::VectorXd y_pred = model.predict(X_test);
        
        // Calculate metrics
        double mse = calculateMSE(y_test, y_pred);
        double rmse = sqrt(mse);
        
        // Calculate R-squared
        double ss_tot = (y_test.array() - y_test.mean()).square().sum();
        double ss_res = (y_test - y_pred).array().square().sum();
        double r2 = 1.0 - (ss_res / ss_tot);
        
        // Count non-zero coefficients
        int non_zero = (model.get_coefficients().array().abs() > 1e-6).count();
        
        std::cout << "  Non-zero coefficients: " << non_zero << "/" << n_features << "\n";
        std::cout << "  MSE: " << mse << "\n";
        std::cout << "  RMSE: " << rmse << "\n";
        std::cout << "  R-squared: " << r2 << "\n\n";
    }
    
    // Show coefficient comparison for lambda=10
    std::cout << "Coefficient comparison (lambda=10):\n";
    std::cout << "==================================\n";
    LassoReg final_model(10.0, 1000, 1e-6);
    final_model.fit(X_train, y_train);
    
    std::cout << "Feature | True Coef | Estimated Coef\n";
    std::cout << "--------|-----------|---------------\n";
    for (int i = 0; i < std::min(10, n_features); ++i) {
        std::cout << "   " << i << "    | " 
                  << std::fixed << std::setprecision(4) << std::setw(9) << true_params(i) 
                  << " | " << std::setw(14) << final_model.get_coefficients()(i) << "\n";
    }
    if (n_features > 10) {
        std::cout << "   ...  |    ...    |      ...\n";
    }

    return 0;
}
