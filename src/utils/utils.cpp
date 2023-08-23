#include <iostream>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include <Layers/Layers.hpp>

Eigen::VectorXd predict(const std::vector<Layer*>& network, const Eigen::VectorXd& input) {
    Eigen::VectorXd output = input;
    for (const auto& layer : network) {
        layer->forward(output);
        output = layer->getOutput();
    }
    return output;
}

void train(std::vector<Layer*>& network,
           const std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>& loss,
           const std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>& loss_prime,
           const std::vector<Eigen::VectorXd>& x_train,
           const std::vector<Eigen::VectorXd>& y_train,
           int epochs = 1000,
           double learning_rate = 0.01,
           bool verbose = true) {

    for (int e = 0; e < epochs; ++e) {
        double total_error = 0.0;
        for (size_t i = 0; i < x_train.size(); ++i) {
            // Forward pass
            Eigen::VectorXd output = predict(network, x_train[i]);

            // Compute loss
            Eigen::VectorXd error = loss(y_train[i], output);
            total_error += error.sum();

            // Backward pass
            Eigen::VectorXd grad = loss_prime(y_train[i], output);
            for (auto it = network.rbegin(); it != network.rend(); ++it) {
                (*it)->backward(grad, learning_rate);
                grad = (*it)->getInputGradient();
            }
        }

        double average_error = total_error / x_train.size();
        if (verbose) {
            std::cout << e + 1 << "/" << epochs << ", error=" << average_error << std::endl;
        }
    }
}