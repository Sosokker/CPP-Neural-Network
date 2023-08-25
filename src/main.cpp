#include <iostream>
#include <vector>
#include "../include/Eigen/Dense"
#include "Activation/ActivationFunc.cpp"
#include "Loss/Loss.cpp"
#include "Layers/DenseLayer.hpp"
#include "utils/utils.hpp"

int main() {

    std::vector<Layer*> network;
    network.push_back(new DenseLayer(2, 4));
    network.push_back(new TanhActivation());
    network.push_back(new DenseLayer(4, 1));
    network.push_back(new SigmoidActivation());

    std::vector<Eigen::VectorXd> x_train;
    std::vector<Eigen::VectorXd> y_train;

    // ! SAMPLE
    Eigen::VectorXd x_sample1(2);
    x_sample1 << 0.2, 0.3;
    x_train.push_back(x_sample1);

    Eigen::VectorXd y_sample1(1);
    y_sample1 << 1.0; // Positive class
    y_train.push_back(y_sample1);

    Eigen::VectorXd x_sample2(2);
    x_sample2 << -0.5, 0.8;
    x_train.push_back(x_sample2);

    Eigen::VectorXd y_sample2(1);
    y_sample2 << 0.0; // Negative class
    y_train.push_back(y_sample2);

    Eigen::VectorXd x_sample3(2);
    x_sample3 << 0.7, -0.2;
    x_train.push_back(x_sample3);

    Eigen::VectorXd y_sample3(1);
    y_sample3 << 1.0; // Positive class
    y_train.push_back(y_sample3);

    Eigen::VectorXd x_sample4(2);
    x_sample4 << -0.8, -0.5;
    x_train.push_back(x_sample4);

    Eigen::VectorXd y_sample4(1);
    y_sample4 << 0.0; // Negative class
    y_train.push_back(y_sample4);

    Eigen::VectorXd x_sample5(2);
    x_sample5 << 0.9, 0.1;
    x_train.push_back(x_sample5);

    Eigen::VectorXd y_sample5(1);
    y_sample5 << 1.0; // Positive class
    y_train.push_back(y_sample5);

    Eigen::VectorXd x_sample6(2);
    x_sample6 << -0.3, 0.6;
    x_train.push_back(x_sample6);

    Eigen::VectorXd y_sample6(1);
    y_sample6 << 0.0; // Negative class
    y_train.push_back(y_sample6);

    Eigen::VectorXd x_sample7(2);
    x_sample7 << 0.5, -0.7;
    x_train.push_back(x_sample7);

    Eigen::VectorXd y_sample7(1);
    y_sample7 << 1.0; // Positive class
    y_train.push_back(y_sample7);

    Eigen::VectorXd x_sample8(2);
    x_sample8 << -0.1, 0.4;
    x_train.push_back(x_sample8);

    Eigen::VectorXd y_sample8(1);
    y_sample8 << 0.0; // Negative class
    y_train.push_back(y_sample8);

    Eigen::VectorXd x_sample9(2);
    x_sample9 << 0.6, 0.0;
    x_train.push_back(x_sample9);

    Eigen::VectorXd y_sample9(1);
    y_sample9 << 1.0; // Positive class
    y_train.push_back(y_sample9);

    Eigen::VectorXd x_sample10(2);
    x_sample10 << -0.6, -0.3;
    x_train.push_back(x_sample10);

    Eigen::VectorXd y_sample10(1);
    y_sample10 << 0.0; // Negative class
    y_train.push_back(y_sample10);

    auto loss_function = binary_cross_entropy;
    auto loss_prime_function = binary_cross_entropy_prime;

    // Train
    int epochs = 1000;
    double learning_rate = 0.01;
    train(network, loss_function, loss_prime_function, x_train, y_train, epochs, learning_rate, true);

    //* Use the trained model for predictions
    Eigen::VectorXd input_to_predict;
    Eigen::VectorXd predicted_output = predict(network, input_to_predict);

    std::cout << "Predicted Output: " << predicted_output.transpose() << std::endl;

    // ! Clean up: Delete the layers in the network
    for (auto layer : network) {
        delete layer;
    }

    return 0;
}
