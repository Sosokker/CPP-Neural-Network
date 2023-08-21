#include "Loss.hpp"

double Loss::regularization_loss() {
    // Implementation
}

// Implement other member functions of Loss class

double Loss_CategoricalCrossentropy::forward(const std::vector<double>& y_pred, const std::vector<double>& y_true) {
    // Implementation
}

void Loss_CategoricalCrossentropy::backward(std::vector<double>& dvalues, const std::vector<double>& y_true) {
    // Implementation
}

double Loss_BinaryCrossentropy::forward(const std::vector<double>& y_pred, const std::vector<double>& y_true) {
    // Implementation
}

void Loss_BinaryCrossentropy::backward(std::vector<double>& dvalues, const std::vector<double>& y_true) {
    // Implementation
}

double Loss_MeanSquaredError::forward(const std::vector<double>& y_pred, const std::vector<double>& y_true) {
    // Implementation
}

void Loss_MeanSquaredError::backward(std::vector<double>& dvalues, const std::vector<double>& y_true) {
    // Implementation
}

double Loss_MeanAbsoluteError::forward(const std::vector<double>& y_pred, const std::vector<double>& y_true) {
    // Implementation
}

void Loss_MeanAbsoluteError::backward(std::vector<double>& dvalues, const std::vector<double>& y_true) {
    // Implementation
}
