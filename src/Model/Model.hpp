#pragma once
#include "ActivationFunction/ActivationFunction.hpp"
#include "Loss/Loss.hpp"
#include "Optimizer/Optimizer.hpp"
#include "Accuracy/Accuracy.hpp"
#include "Layers/Layers.hpp"

class Model {
private:
    Loss* loss;
    Optimizer* optimizer;
    Accuracy* accuracy;
    std::vector<Layer*> layers;

public:
    Model();

    void add(Layer& layer);
    void set_loss(Loss& loss);
    void set_optimizer(Optimizer& optimizer);
    void set_accuracy(Accuracy& accuracy);
    void finalize();

    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y,
               int epochs, int batch_size, int print_every, const std::vector<std::vector<double>>& validation_data);

    void evaluate(const std::vector<std::vector<double>>& X_val, const std::vector<std::vector<double>>& y_val, int batch_size);

    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& X, int batch_size);

    void save_parameters(const std::string& path);
    void load_parameters(const std::string& path);
    void save(const std::string& path);

    void forward(const std::vector<std::vector<double>>& X, bool training);
    void backward(const std::vector<std::vector<double>>& output, const std::vector<std::vector<double>>& y);

    std::vector<std::vector<std::vector<double>>> get_parameters();
    void set_parameters(const std::vector<std::vector<std::vector<double>>>& parameters);

    static Model load(const std::string& path);
};
