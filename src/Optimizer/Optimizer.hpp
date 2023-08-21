#pragma once
#include "Layers/Layers.hpp"

class Optimizer {
public:
    virtual void pre_update_params() = 0;
    virtual void update_params(Layer& layer) = 0;
    virtual void post_update_params() = 0;
    // Other common members and methods
};

class Optimizer_SGD : public Optimizer {
public:
    Optimizer_SGD(double learning_rate, double decay, double momentum);

    void pre_update_params() override;
    void update_params(Layer& layer) override;
    void post_update_params() override;

private:
    double learning_rate;
    double current_learning_rate;
    double decay;
    int iterations;
    double momentum;
};

class Optimizer_Adagrad : public Optimizer {
public:
    Optimizer_Adagrad(double learning_rate, double decay, double epsilon);

    void pre_update_params() override;
    void update_params(Layer& layer) override;
    void post_update_params() override;

private:
    double learning_rate;
    double current_learning_rate;
    double decay;
    int iterations;
    double epsilon;
};

// Similar declarations for Optimizer_RMSprop and Optimizer_Adam

