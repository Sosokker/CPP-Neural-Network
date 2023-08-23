#include "DenseLayer.hpp"

DenseLayer::DenseLayer(int input_size, int output_size) {
    weights = Eigen::MatrixXd::Random(output_size, input_size);
    bias = Eigen::VectorXd::Random(output_size);
}

void DenseLayer::forward(const Eigen::VectorXd& input) {
    this->input = input;
    output = weights * input + bias;
}

void DenseLayer::backward(const Eigen::VectorXd& output_gradient, double learning_rate) {
    Eigen::MatrixXd weights_gradient = output_gradient * input.transpose();
    input_gradient = weights.transpose() * output_gradient;
    weights -= learning_rate * weights_gradient;
    bias -= learning_rate * output_gradient;
}

const Eigen::VectorXd& DenseLayer::getOutput() const {
    return output;
}

const Eigen::VectorXd& DenseLayer::getInputGradient() const {
    return input_gradient;
}
