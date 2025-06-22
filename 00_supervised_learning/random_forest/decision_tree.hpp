#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <map>
#include <memory> // For std::unique_ptr
#include <algorithm> // For std::shuffle
#include <random>    // For std::mt19937 and std::uniform_int_distribution


/**
 * @brief class for specifying the type of task for the decision tree.
 * 
 */
enum class TaskType {
    CLASSIFICATION,
    REGRESSION  
};


/**
 * @brief Represents a single data point as features and target.
 * 
 */
struct DataPoint {
    std::vector<double> features;
    double target; //discrete (class.) or continuous (regr.)
}


/**
 * @brief 
 */
struct Node {
    int feature_index = -1;            // Index of the feature to split on
    double threshold = 0.0;            //Threshold value for the split
    double predicted_value = 0.0;      // Predicted value if leaf node
    std::unique_ptr<Node> left;        // Left child (if feature < threshold)
    std::unique_ptr<Node> right;       // Right child (if feature >= threshold)
    bool is_leaf = false;              // True if this node is a leaf node

    Node() = default;
}

class DecisionTree {
public:
    DecisionTree(TaskType task_type = TaskType::CLASSIFICATION,
                int max_depth = 6,
                int min_samples_split = 2,
                double min_impurity_decrease = 0.01,
                int max_features = -1);
    
    /**
     * @brief Train the decision tree model on the provided data.
     * 
     * @param data Data points to train on, each with features and target.
     */
    void train(const std::vector<DataPoint>& data);


    /**
     * @brief Predict the target value for a given set of features.
     * 
     * @param features Features of the data point to predict.
     * @return double Predicted value for the given features.
     */
    double predict(const std::vector<double>& features) const;
private:
    TaskType task_type_;
    std::unique_ptr<Node> root_;
    int max_depth_;
    int min_samples_split_;
    double min_impurity_decrease_;
    int max_features_; // Nr. of features considered at each split (-1 for all)
    std::mt19937 random_generator_;

    /**
     * @brief Build the decision tree recursively.
     * 
     * @param data Data points to build the tree from.
     * @param depth Current depth of the tree.
     * @return std::unique_ptr<Node> Pointer to the root node of the subtree.
     */
    std::unique_ptr<Node> build_tree(
        const std::vector<DataPoint>& data,
        int depth
    );

    /**
     * @brief Find the best feature and threshold to split the data.
     * 
     * @param data Data points to find the best split for.
     * @param feature_indices Indices of features to consider for splitting.
     * @return std::pair<int, double> Index of the best feature and the 
     *          threshold value.
     */
    std::pair<int, double> find_best_split(
        const std::vector<DataPoint>& data,
        const std::vector<int>& feature_indices
    );

    /**
     * @brief Calculate the impurity of the data.
     * 
     * @param data Data points to calculate impurity for.
     * @return double Impurity value.
     */
    double calculate_impurity(const std::vector<DataPoint>& data);

    /**
     * @brief Calculate Gini impurity for classification tasks.
     * 
     * @param data Data points to calculate Gini impurity for.
     * @return double Gini impurity value.
     */
    double calculate_gini_impurity(const std::vector<DataPoint>& data);

    /**
     * @brief Calculate Mean Squared Error (MSE) for regression tasks.
     * 
     * @param data Data points to calculate MSE for.
     * @return double MSE value.
     */
    double calculate_mse(const std::vector<DataPoint>& data);

    /**
     * @brief Split the data into two subsets based on a feature and threshold.
     * 
     * @param data Data points to split.
     * @param feature_index Index of the feature to split on.
     * @param threshold Threshold value for the split.
     * @return std::pair<std::vector<DataPoint>, std::vector<DataPoint>> 
     *          Two subsets of data points (left and right).
     */
    std::pair<std::vector<DataPoint>, std::vector<DataPoint>> split_data(
        const std::vector<DataPoint>& data,
        int feature_index,
        double threshold
    )

    /**
     * @brief Calculate the prediction for a set of data points.
     * 
     * @param data Data points to calculate the prediction for.
     * @return double Predicted value based on the data.
     */
    double calculate_prediction(
        const std::vector<DataPoint>& data
    );

    /**
     * @brief Perform majority vote for classification tasks.
     * 
     * @param data Data points to perform majority vote on.
     * @return int Predicted class label.
     */
    int majority_vote(const std::vector<DataPoint>& data);

    /**
     * @brief Calculate the mean value for regression tasks.
     * 
     * @param data Data points to calculate the mean for.
     * @return double Mean value of the target variable.
     */
    double calculate_mean(const std::vector<DataPoint>& data);

    /**
     * @brief Get a random subset of feature indices for splitting.
     * 
     * @param total_features Total number of features available.
     * @return std::vector<int> Randomly selected feature indices.
     */
    std::vector<int> get_random_feature_indices(int total_features);
}



#endif // DECISION_TREE_H