#include <Eigen/Dense>
#include "Layers.hpp"

class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size);

    void forward(const Eigen::VectorXd& input) override;
    void backward(const Eigen::VectorXd& output_gradient, double learning_rate) override;

    const Eigen::VectorXd& getOutput() const;
    const Eigen::VectorXd& getInputGradient() const;

private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd bias;
};
