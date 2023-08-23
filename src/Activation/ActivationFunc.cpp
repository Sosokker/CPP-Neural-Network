#include <cmath>
#include <Layers/Layers.hpp>
#include "Activation.hpp"
#include <Eigen/Dense>

class TanhActivation : public Activation {
public:
    TanhActivation() : Activation(tanh, tanh_prime) {}

private:
    static Eigen::VectorXd tanh(const Eigen::VectorXd& x) {
        return x.array().tanh();
    }

    static Eigen::VectorXd tanh_prime(const Eigen::VectorXd& x) {
        return 1 - x.array().tanh().square();
    }
};

class SigmoidActivation : public Activation {
public:
    SigmoidActivation() : Activation(sigmoid, sigmoid_prime) {}

private:
    static Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) {
        return 1.0 / (1.0 + (-x.array()).exp());
    }

    static Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd& x) {
        Eigen::VectorXd s = sigmoid(x);
        return s.array() * (1.0 - s.array());
    }
};