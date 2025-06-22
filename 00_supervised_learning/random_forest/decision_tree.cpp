#include "decision_tree.hpp"

#include <iostream>
#include <numeric>  // For std::iota
#include <set>
#include <limits>   // For std::numeric_limits

DecisionTree::DecisionTree(
    TaskType task_type,
    int max_depth,
    int min_samples_split,
    double min_impurity_decrease,
    int max_features
) : task_type_(task_type),
    max_depth_(max_depth),
    min_samples_split_(min_samples_split),
    min_impurity_decrease_(min_impurity_decrease),
    max_features_(max_features),
    random_generator_(std::random_device{}()
)


/**
 * @brief Train the decision tree model on the provided data.
 * 
 * @param data Data points to train on, each with features and target.
 */
void DecisionTree::train(const std::vector<DataPoint>& data)
{
    //raise an error if data is empty
    if (data.empty()) {
        throw std::invalid_argument("Training data cannot be empty.");
    root_ = build_tree(data, 0);
}


/**
 * @brief Predict the target value for a given set of features.
 * 
 * @param features Features of the data point to predict.
 * @return double Predicted value for the given features.
 */
double DecisionTree::predict(
    const std::vector<double>& features
) const
{
    Node* current_node = root_.get();
    while (current_node && !current_node -> is_leaf)
    {
        if (features[current_node -> feature_index] < current_node -> threshold)
        {
            current_node = current_node->left.get();
        }
        else
        {
            current_node = current_node->right.get();
        }
    }
    return current_node ? current_node->predicted_value
                         : 0.0;  // Should not happen with a trained tree
}

/**
 * @brief Build the decision tree recursively.
 * 
 * @param data Data points to build the tree from.
 * @param depth Current depth of the tree.
 * @return std::unique_ptr<Node> Pointer to the root node of the subtree.
 */
std::unique_ptr<Node> DecisionTree::build_tree(
    const std::vector<DataPoint>& data,
    int depth
)
{
    std::unique_ptr<Node> node = std::make_unique<Node>();

    //Stop conditions
    if (data.empty())
    {
        node->is_leaf = true;
        node->predicted_value = 0.0; //default value
        return node;
    }

    double current_prediction = calculate_prediction(data);
    // Check for stopping conditions
    if (depth >= max_depth_ ||
        static_cast<int>(data.size()) < min_samples_split_ ||
        calculate_impurity(data) == 0.0)
    {  // Pure node
        node->is_leaf = true;
        node->predicted_value = current_prediction;
        return node;
    }

    std::vector<int> feature_indices;
    if (!data.empty() && !data[0].features.empty())
    {
        feature_indices = get_random_feature_indices(data[0].features.size());
    }
    else
    {  // No features to split on
        node->is_leaf = true;
        node->predicted_value = current_prediction;
        return node;
    }

    auto best_split = find_best_split(data, feature_indices);
    int best_feature_index = best_split.first;
    double best_threshold = best_split.second;

    double current_impurity = calculate_impurity(data);
    auto [left_data, right_data] = split_data(
        data, best_feature_index, best_threshold
    );

    if (left_data.empty() || right_data.empty())
    {  // Could not split further
        node->is_leaf = true;
        node->predicted_value = current_prediction;
        return node;
    }

    double left_impurity = calculate_impurity(left_data);
    double right_impurity = calculate_impurity(right_data);
    double N = data.size();
    double N_left = left_data.size();
    double N_right = right_data.size();
    double weighted_impurity =
        (N_left / N) * left_impurity + (N_right / N) * right_impurity;
    double impurity_decrease = current_impurity - weighted_impurity;
    if (impurity_decrease < min_impurity_decrease_ || best_feature_index == -1)
    {
        // No sufficient impurity decrease found
        node->is_leaf = true;
        node->predicted_value = current_prediction;
        return node;
    }
    node->feature_index = best_feature_index;
    node->threshold = best_threshold;
    node->left = build_tree(left_data, depth + 1);
    node->right = build_tree(right_data, depth + 1);

    return node;
}

/**
 * @brief Split the data into two subsets based on a feature and threshold.
 * 
 * @param data Data points to split.
 * @param feature_index Index of the feature to split on.
 * @param threshold Threshold value for the split.
 * @return std::pair<std::vector<DataPoint>, std::vector<DataPoint>> 
 *          Two subsets of data points (left and right).
 */
std::pair<std::vector<DataPoint>, std::vector<DataPoint>>
DecisionTree::split_data(const std::vector<DataPoint>& data,
                         int feature_index,
                         double threshold)
{
    std::vector<DataPoint> left_data;
    std::vector<DataPoint> right_data;

    if (feature_index < 0) return {data, {}};  // Or handle error

    for (const auto& point : data)
    {
        if (point.features[feature_index] < threshold)
        {
            left_data.push_back(point);
        }
        else
        {
            right_data.push_back(point);
        }
    }
    return {left_data, right_data};
}

/**
 * @brief Split the data into two subsets based on a feature and threshold.
 * 
 * @param data Data points to split.
 * @param feature_index Index of the feature to split on.
 * @param threshold Threshold value for the split.
 * @return std::pair<std::vector<DataPoint>, std::vector<DataPoint>> 
 *          Two subsets of data points (left and right).
 */
std::pair<int, double> DecisionTree::find_best_split(
    const std::vector<DataPoint>& data,
    const std::vector<int>& feature_indices
)
{
    if (data.empty() || feature_indices.empty())
    {
        return {-1, 0.0}; // No valid split
    }

    double best_weighted_impurity = std::numeric_limits<double>::max();
    int best_feature_index = -1;
    double best_threshold = 0.0;

    double initial_impurity = calculate_impurity(data);

    for (int feature_idx : feature_indices)
    {
        // get unique values for the feature
        std::set<double> unique_values;
        for (const auto& point : data)
        {
            unique_values.insert(point.features[feature_idx]);
        }

        if (unique_values.size() < 2)
        { 
            // Cannot split on feature if only one unique value
            continue;
        }

        std::vector<double> sorted_thresholds(
            unique_values.begin(), unique_values.end()
        )

        for (size_t i=0; i < sorted_thresholds.size() - 1; ++i)
        {
            // Midpoint as threshold
            double threshold = 
            (sorted_thresholds[i] + sorted_thresholds[i + 1]) / 2.0; 
            auto [left_data, right_data] = split_data(
                data, feature_idx, threshold
            )

            if (left_data.empty() || right_data.empty())
            {
                continue; // Cannot split if one side is empty
            }

            double p_left = 
                static_cast<double>(left_data.size()) / data.size();
            double p_right = 
                static_cast<double>(right_data.size()) / data.size();
            double current_impurity_weighted =
                p_left * calculate_impurity(left_data) +
                p_right * calculate_impurity(right_data);
            
            if (current_impurity_weighted < best_weighted_impurity)
            {
                best_weighted_impurity = current_impurity_weighted;
                best_feature_index = feature_idx;
                best_threshold = threshold; 
            }
        }
    }

    // If no split improves impurity significantly, return -1
    // will then become a leaf node
    if (best_feature_index != -1 &&
        (initial_impurity - best_weighted_impurity) < min_impurity_decrease_
        && initial_impurity > 0)
    {
        // No sufficient impurity decrease found
        return {-1, 0.0};
    }

    return {best_feature_index, best_threshold};
}

/**
 * @brief Calculate the prediction for a set of data points.
 * 
 * @param data Data points to calculate the prediction for.
 * @return double Predicted value based on the data.
 */
double DecisionTree::calculate_prediction(
    const std::vector<DataPoint>& data
)
{
    if (task_type_ == TaskType::CLASSIFICATION)
    {
        return static_cast<double>(majority_vote(data));
    }
    else
    {
        return calculate_mean(data);
    }
}

int DecisionTree::majority_vote(const std::vector<DataPoint>& data)
{
    if (data.empty())
    {
        throw std::invalid_argument("Data cannot be empty for majority vote.");
    }
    std::map<int, int> label_counts;
    for (const auto& point : data)
    {
        label_counts[static_cast<int>(point.target)]++;
    }

    int majority_label = -1;
    int max_count = 0;
    for (const auto& pair : label_counts)
    {
        if (pair.second > max_count)
        {
            max_count = pair.second;
            majority_label = pair.first;
        }
    }
    return majority_label;
}


/**
 * @brief Calculate the impurity of the data.
 * 
 * @param data Data points to calculate impurity for.
 * @return double Impurity value.
 */
double DecisionTree::calculate_impurity(
    const std::vector<DataPoint>& data
)
{
    if (task_type_ == TaskType::CLASSIFICATION)
    {
        return calculate_gini_impurity(data);
    }
    else
    {
        return calculate_mse(data);
    }
}




/**
 * @brief Calculate Gini impurity for classification tasks.
 * 
 * @param data Data points to calculate Gini impurity for.
 * @return double Gini impurity value.
 */
double DecisionTree::calculate_gini_impurity(
    const std::vector<DataPoint>& data
)
{
    if (data.empty())
    {
        return 0.0; // No data to calculate Gini impurity
    }

    std::map<int, int> label_counts;

    for (const auto& point : data)
    {
        // Convert target (double) to int to get class label
        int label = static_cast<int>(point.target);
        
        // Increment count for this label
        label_counts[label]++;
    }

    double impurity = 1.0;
    for (const auto& pair : label_counts)
    {
        // Proportion of each class
        double p_i = static_cast<double>(pair.second) / data.size();
        impurity -= p_i * p_i;
    }
    return impurity;
}



/**
 * @brief Calculate Gini impurity for classification tasks.
 * 
 * @param data Data points to calculate Gini impurity for.
 * @return double Gini impurity value.
 */
double DecisionTree::calculate_mse(
    const std::vector<DataPoint>& data
)
{
    if (data.empty())
    {
        return 0.0; // No data to calculate MSE
    }

    double mean = calculate_mean(data);
    double mse = 0.0;
    
    for (const auto& point : data)
    {
        double diff = point.target - mean;
        mse += diff * diff;
    }
    return mse / data.size();
}


/**
 * @brief Calculate the prediction for a set of data points.
 * 
 * @param data Data points to calculate the prediction for.
 * @return double Predicted value based on the data.
 */
double DecisionTree::calculate_mean(
    const std::vector<DataPoint>& data
)
{
    if (data.empty())
    {
        return 0.0; // No data to calculate mean
    }

    double sum = 0.0;
    for (const auto& point : data)
    {
        sum += point.target;
    }
    return sum / data.size();
}

/**
 * @brief Get a random subset of feature indices for splitting.
 * 
 * @param total_features Total number of features available.
 * @return std::vector<int> Randomly selected feature indices.
 */
std::vector<int> DecisionTree::get_random_feature_indices(int total_features)
{
    std::vector<int> all_indices(total_features);
    std::iota(
        all_indices.begin(), all_indices.end(), 0
    );

    if (max_features_ <= 0 || max_features_ >= total_features)
    {
        return all_indices; // Use all features
    }

    std::shuffle(all_indices.begin(), all_indices.end(), random_generator_);
    return std::vector<int>(
        all_indices.begin(),
        all_indices.begin() + max_features_
    );
}