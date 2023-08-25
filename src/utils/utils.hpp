#ifndef NEURAL_NETWORK_UTILS_HPP
#define NEURAL_NETWORK_UTILS_HPP

#include <vector>
#include <functional>
#include "../../include/Eigen/Dense"
#include "../Layers/Layers.hpp"

Eigen::VectorXd predict(const std::vector<Layer*>& network, const Eigen::VectorXd& input);

void train(std::vector<Layer*>& network,
           const std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>& loss,
           const std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>& loss_prime,
           const std::vector<Eigen::VectorXd>& x_train,
           const std::vector<Eigen::VectorXd>& y_train,
           int epochs = 1000,
           double learning_rate = 0.01,
           bool verbose = true);

#endif // NEURAL_NETWORK_UTILS_HPP
