#include "Optimizer/Optimizer.hpp"
#include "Layers/Layers.hpp"

Optimizer_SGD::Optimizer_SGD(double learning_rate, double decay, double momentum)
    : learning_rate(learning_rate),
      current_learning_rate(learning_rate),
      decay(decay),
      iterations(0),
      momentum(momentum) {}

void Optimizer_SGD::pre_update_params() {
    // pre_update_params implementation
}

void Optimizer_SGD::update_params(Layer& layer) {
    // update_params implementation for any Layer
}

void Optimizer_SGD::post_update_params() {
    iterations++;
}

Optimizer_Adagrad::Optimizer_Adagrad(double learning_rate, double decay, double epsilon)
    : learning_rate(learning_rate),
      current_learning_rate(learning_rate),
      decay(decay),
      iterations(0),
      epsilon(epsilon) {}

void Optimizer_Adagrad::pre_update_params() {
    // pre_update_params implementation
}

void Optimizer_Adagrad::update_params(Layer& layer) {
    // update_params implementation for any Layer
}

void Optimizer_Adagrad::post_update_params() {
    iterations++;
}

// Similar implementations for Optimizer_RMSprop and Optimizer_Adam

