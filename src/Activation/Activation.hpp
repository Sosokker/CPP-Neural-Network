#include <functional>
#include <Eigen/Dense>
#include <Layers/Layers.hpp>

class Activation : public Layer {
public:
    Activation(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation,
               std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_prime);

    void forward(const Eigen::VectorXd& input) override;
    void backward(const Eigen::VectorXd& output_gradient, double learning_rate) override;

private:
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_prime;
};
