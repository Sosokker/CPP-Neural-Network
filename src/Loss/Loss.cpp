#include "../../include/Eigen/Dense"
#include <cmath>

Eigen::VectorXd mse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    return (y_true - y_pred).array().square() / y_true.size();
}

Eigen::VectorXd mse_prime(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    return 2.0 * (y_pred - y_true) / y_true.size();
}

Eigen::VectorXd binary_cross_entropy(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    return -((y_true.array() * y_pred.array().log()) + ((1 - y_true.array()) * (1 - y_pred.array()).log())) / y_true.size();
}

Eigen::VectorXd binary_cross_entropy_prime(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    return ((1 - y_true.array()) / (1 - y_pred.array()) - y_true.array() / y_pred.array()) / y_true.size();
}
