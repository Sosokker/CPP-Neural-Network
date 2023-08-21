#ifndef LOSS_HPP
#define LOSS_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include "Layers/Layers.hpp"

class Loss {
public:
    virtual double regularization_loss();
    virtual double forward(const std::vector<double>& y_pred, const std::vector<double>& y_true);
    virtual void backward(std::vector<double>& dvalues, const std::vector<double>& y_true);
    virtual void remember_trainable_layers(const std::vector<Layer*>& trainable_layers);
    virtual double calculate(const std::vector<double>& output, const std::vector<double>& y, bool include_regularization = false);
    virtual double calculate_accumulated(bool include_regularization = false);
    virtual void new_pass();

private:
    std::vector<Layer*> trainable_layers;
    double accumulated_sum;
    int accumulated_count;
};

class Loss_CategoricalCrossentropy : public Loss {
public:
    double forward(const std::vector<double>& y_pred, const std::vector<double>& y_true) override;
    void backward(std::vector<double>& dvalues, const std::vector<double>& y_true) override;
};

class Loss_BinaryCrossentropy : public Loss {
public:
    double forward(const std::vector<double>& y_pred, const std::vector<double>& y_true) override;
    void backward(std::vector<double>& dvalues, const std::vector<double>& y_true) override;
};

class Loss_MeanSquaredError : public Loss {
public:
    double forward(const std::vector<double>& y_pred, const std::vector<double>& y_true) override;
    void backward(std::vector<double>& dvalues, const std::vector<double>& y_true) override;
};

class Loss_MeanAbsoluteError : public Loss {
public:
    double forward(const std::vector<double>& y_pred, const std::vector<double>& y_true) override;
    void backward(std::vector<double>& dvalues, const std::vector<double>& y_true) override;
};

#endif // LOSS_HPP
