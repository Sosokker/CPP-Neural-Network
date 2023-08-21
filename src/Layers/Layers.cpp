#include "Layers.hpp"
#include <random>

Layer_Dense::Layer_Dense(int n_inputs, int n_neurons,
                double weight_regularizer_l1,
                double weight_regularizer_l2,
                double bias_regularizer_l1,
                double bias_regularizer_l2)
    : weight_regularizer_l1(weight_regularizer_l1),
      weight_regularizer_l2(weight_regularizer_l2),
      bias_regularizer_l1(bias_regularizer_l1),
      bias_regularizer_l2(bias_regularizer_l2)
{
    // Initialize weights and biases
    weights.resize(n_inputs, std::vector<double>(n_neurons));
    for (auto& row : weights) {
        for (double& weight : row) {
            weight = 0.01 * (std::rand() / double(RAND_MAX));
        }
    }
    biases.resize(1, std::vector<double>(n_neurons, 0.0));
}

void Layer_Dense::forward(const std::vector<double>& inputs, bool training) {
    this->inputs = inputs;

    // Forward pass implementation
}

void Layer_Dense::backward(std::vector<double>& dvalues) {
    // Backward pass implementation
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> Layer_Dense::get_parameters() const {
    return std::make_pair(weights, biases);
}

void Layer_Dense::set_parameters(const std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& biases) {
    this->weights = weights;
    this->biases = biases;
}

Layer_Dropout::Layer_Dropout(double rate) : rate(1 - rate) {}

void Layer_Dropout::forward(const std::vector<double>& inputs, bool training) {    // Forward pass implementation
}

void Layer_Dropout::backward(std::vector<double>& dvalues) {
    // Backward pass implementation
}

void Layer_Input::forward(const std::vector<double>& inputs, bool training) {
    this->output = inputs;
}

// Implement similar member functions for other Layer classes
