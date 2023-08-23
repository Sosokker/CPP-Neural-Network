#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <Eigen/Dense>

class Layer {
public:
    Layer();

    virtual void forward(const Eigen::VectorXd& input_data);
    virtual void backward(const Eigen::VectorXd& output_gradient, double learning_rate);

    const Eigen::VectorXd& getOutput() const;
    const Eigen::VectorXd& getInputGradient() const;

protected:
    Eigen::VectorXd input;
    Eigen::VectorXd output;
    Eigen::VectorXd input_gradient;
};

#endif // LAYERS_HPP
